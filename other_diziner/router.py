"""OpenRouter client with durable request reservations and validated response caching."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import httpx
from dotenv import find_dotenv, load_dotenv


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def write_text(path, text):
    """Atomic replacement; persist reservations before a potentially billable request."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path, value):
    write_text(path, dumps(value) + '\n')


def parse_json(text):
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[-1].strip() != '```':
            raise ValueError('Unclosed JSON fence')
        text = '\n'.join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


class Router:
    def __init__(self, output, max_requests=100, max_cost_usd=3.0, max_tokens=8000, attempts=6):
        load_dotenv(find_dotenv(usecwd=True))
        key = os.getenv('OPENROUTER_API_KEY')
        if not key:
            raise ValueError('Set OPENROUTER_API_KEY in the repo-root .env')
        self.base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1').rstrip('/')
        self.http = httpx.Client(base_url=self.base_url + '/', timeout=120,
                                 headers={'Authorization': f'Bearer {key}',
                                          'X-OpenRouter-Title': 'learning-ai-components DiZiNER'})
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_requests, self.max_cost_usd = max_requests, max_cost_usd
        self.max_tokens, self.attempts = max_tokens, attempts
        self.reasoning = os.getenv('OPENROUTER_REASONING_EFFORT', 'low')
        # One atomic file per request. A pending request remains charged on restart.
        self.records = [json.loads(p.read_text()) for p in sorted((self.output / 'requests').glob('*.json'))]
        legacy = self.output / 'requests.jsonl'
        if legacy.exists():
            self.records.extend(json.loads(line) for line in legacy.read_text().splitlines())
        self.catalog = {}
        self.cache_hits = 0

    def preflight(self, models):
        response = self.http.get('models')
        response.raise_for_status()
        catalog = {row['id']: row for row in response.json()['data']}
        missing = sorted(set(models) - set(catalog))
        if missing:
            raise ValueError(f'Models unavailable on OpenRouter: {missing}. Edit config; no automatic substitutes.')
        found = {m: catalog[m] for m in models}
        path = self.output / 'model_catalog.json'
        if path.exists():
            previous = json.loads(path.read_text())
            # Preserve the decoding-parameter contract for exact resume behavior.
            for model in models:
                if model in previous and set(previous[model].get('supported_parameters', [])) != set(found[model].get('supported_parameters', [])):
                    raise ValueError(f'Decoding support changed for {model}; use a new output directory')
            found = {**previous, **found}
        self.catalog = found
        write_json(path, self.catalog)

    def _reserve(self, record):
        record = {**record, 'request_id': uuid.uuid4().hex, 'status': 'pending'}
        write_json(self.output / 'requests' / f"{record['request_id']}.json", record)
        self.records.append(record)
        return record

    def _settle(self, record, **fields):
        settled = {**record, **fields}
        write_json(self.output / 'requests' / f"{record['request_id']}.json", settled)
        record.update(fields)

    def usage(self):
        return {'requests': len(self.records), 'cache_hits_this_session': self.cache_hits,
                'reported_cost_usd': sum(r.get('cost') or 0 for r in self.records),
                'budget_accounted_usd': sum(r.get('accounted_cost', 0) for r in self.records),
                'requests_without_reported_cost': sum(r.get('cost') is None for r in self.records),
                'pending_requests': sum(r.get('status') == 'pending' for r in self.records)}

    def _body(self, model, prompt):
        params = set(self.catalog[model].get('supported_parameters', []))
        body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': self.max_tokens}
        for name, value in {'temperature': 0.0, 'top_p': 1.0, 'repetition_penalty': 1.0,
                            'frequency_penalty': 0.0, 'presence_penalty': 0.0}.items():
            if name in params:
                body[name] = value
        if 'reasoning' in params:
            body['reasoning'] = {'effort': self.reasoning}
        return body

    @staticmethod
    def _validated(response, validate):
        choice = response['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError(f"Incomplete response: {choice.get('finish_reason')}")
        parsed = parse_json(choice['message'].get('content') or '')
        validate(parsed)
        return parsed

    def complete(self, model, prompt, validate, purpose):
        if model not in self.catalog:
            raise ValueError('Call preflight() before sending completions')
        original = self._body(model, prompt)
        cache_id = hashlib.sha256(dumps({'url': self.base_url, 'body': original}).encode()).hexdigest()
        cache = self.output / 'cache' / f'{cache_id}.json'
        invalid = (ValueError, KeyError, IndexError, TypeError, AttributeError)
        if cache.exists():
            saved = json.loads(cache.read_text())
            try:
                parsed = self._validated(saved['response'], validate)
            except invalid:
                pass  # A changed validator must not trap the caller in a rejected cache entry.
            else:
                self.cache_hits += 1
                return parsed
        body, last_error = original, ''
        rejected_answers = set()
        for attempt in range(self.attempts):
            price = self.catalog[model]['pricing']
            reservation = ((len(dumps(body['messages']).encode()) + 256) * float(price['prompt'])
                           + self.max_tokens * float(price['completion']))
            if len(self.records) >= self.max_requests:
                raise RuntimeError('Request budget reached; raise max_requests and resume the same output directory')
            if self.usage()['budget_accounted_usd'] + reservation > self.max_cost_usd:
                raise RuntimeError('Cost budget reached (conservative reservation); raise max_cost_usd to resume')
            record = self._reserve({'purpose': purpose, 'model': model, 'cache_id': cache_id,
                                    'cost': None, 'accounted_cost': reservation})
            print(f"  API {len(self.records)}: {purpose} / {model}", flush=True)
            audit = self.output / 'responses' / f"{record['request_id']}.json"
            write_json(audit, {'request': body, 'status': 'pending'})
            try:
                response = self.http.post('chat/completions', json=body)
            except httpx.TransportError as exc:
                self._settle(record, status=type(exc).__name__)
                if attempt + 1 == self.attempts:
                    raise RuntimeError(f'OpenRouter transport failure: {type(exc).__name__}') from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code != 200:
                self._settle(record, status=response.status_code)
                write_json(audit, {'request': body, 'status': response.status_code, 'raw_response': response.text})
                if response.status_code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == self.attempts:
                    raise RuntimeError(f'OpenRouter HTTP {response.status_code} for {model}; inspect account/model access')
                time.sleep(2 ** attempt)
                continue
            try:
                response_data = response.json()
                if not isinstance(response_data, dict):
                    raise ValueError('API response must be a JSON object')
            except ValueError:
                self._settle(record, status='invalid_response_json')
                write_json(audit, {'request': body, 'status': 200, 'raw_response': response.text})
                last_error = 'API returned malformed response JSON'
                continue
            usage = response_data.get('usage') or {}
            if not isinstance(usage, dict):
                usage = {}
            cost = usage.get('cost')
            if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
                cost = None
            self._settle(record, status=200, cost=cost, accounted_cost=cost if cost is not None else reservation,
                         usage=usage, response_id=response_data.get('id'),
                         served_model=response_data.get('model'), provider=response_data.get('provider'))
            saved = {'initial_request': original, 'request': body, 'response': response_data,
                     'request_id': record['request_id']}
            write_json(audit, saved)
            try:
                parsed = self._validated(response_data, validate)
            except invalid as exc:
                last_error = str(exc)
                self._settle(record, validation_error=last_error)
                # Include the rejected output so the model can repair concrete mistakes.
                # Start from the original task each time; bound context to one rejected answer.
                repair = (f'Correction attempt {attempt + 1}: the previous response was invalid. '
                          + last_error + '\nReturn a complete corrected JSON object matching the original task. '
                          'Do not return a patch or explanatory prose.')
                body = self._body(model, prompt)
                choices = response_data.get('choices')
                rejected = None
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    message = choices[0].get('message')
                    if isinstance(message, dict):
                        rejected = message.get('content')
                repeated = isinstance(rejected, str) and rejected.strip() in rejected_answers
                if repeated:
                    # Replaying an identical bad answer can anchor deterministic models to it.
                    # Keep the original task and concrete validator feedback, but start afresh.
                    body['messages'] = [{'role': 'user', 'content': prompt + '\n\n'
                        + 'A previous attempt repeated an invalid answer. Annotate the input afresh.\n'
                        + repair}]
                else:
                    if isinstance(rejected, str) and rejected.strip():
                        body['messages'].append({'role': 'assistant', 'content': rejected})
                    body['messages'].append({'role': 'user', 'content': repair})
                if isinstance(rejected, str):
                    rejected_answers.add(rejected.strip())
            else:
                # Only accepted responses enter the reusable cache; every failed attempt stays in the audit.
                write_json(cache, saved)
                return parsed
        raise ValueError(f'Invalid {purpose} response after {self.attempts} attempts: {last_error}')

    def close(self):
        self.http.close()
