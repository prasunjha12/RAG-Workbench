"""Run with unittest or pipe into Python inside the API container."""
import json
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi import HTTPException
from openai import OpenAI
from app import provider


class ProviderTests(unittest.TestCase):
    def transport(self, handler):
        return OpenAI(api_key='test-secret', base_url=provider.BASE_URL,
                      max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    def test_missing_key_does_not_use_openai_key(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': '', 'OPENAI_API_KEY': 'legacy'}):
            self.assertFalse(provider.configured())
            with self.assertRaises(HTTPException) as result:
                provider.client()
            self.assertEqual(result.exception.status_code, 503)
            self.assertIn('OPENROUTER_API_KEY', result.exception.detail)

    def test_client_uses_explicit_router_url(self):
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-secret', 'OPENAI_BASE_URL': 'https://wrong.example'}):
            with provider.client() as api:
                self.assertEqual(str(api.base_url), provider.BASE_URL + '/')
                self.assertEqual(api.api_key, 'test-secret')
            self.assertNotIn('test-secret', json.dumps(provider.configuration()))

    def test_embedding_contract_and_order(self):
        def handler(request):
            self.assertEqual(str(request.url), provider.BASE_URL + '/embeddings')
            self.assertEqual(request.headers['Authorization'], 'Bearer test-secret')
            body = json.loads(request.content)
            self.assertEqual(body['model'], provider.EMBED_MODEL)
            self.assertEqual(body['dimensions'], 1536)
            self.assertEqual(body['encoding_format'], 'float')
            self.assertEqual(body['input'], ['one', 'two'])
            return httpx.Response(200, json={'object':'list','model':provider.EMBED_MODEL,
                'data':[{'object':'embedding','index':i,'embedding':[float(i+1)]*1536} for i in [1,0]],
                'usage':{'prompt_tokens':2,'total_tokens':2}})
        with patch.object(provider, 'client', return_value=self.transport(handler)):
            vectors = provider.embed(['one','two'])
        self.assertEqual(vectors[0].to_list()[0], 1.0)
        self.assertEqual(vectors[1].to_list()[0], 2.0)

    def test_reject_wrong_dimensions(self):
        def handler(request):
            return httpx.Response(200,json={'data':[{'index':0,'embedding':[1.0]}]})
        with patch.object(provider, 'client', return_value=self.transport(handler)):
            with self.assertRaises(HTTPException) as result:
                provider.embed(['test'])
        self.assertEqual(result.exception.status_code,502)

    def test_generation_contract(self):
        def handler(request):
            self.assertEqual(str(request.url),provider.BASE_URL+'/chat/completions')
            body=json.loads(request.content)
            self.assertEqual(body['model'],provider.CHAT_MODEL)
            self.assertIn('[1]',body['messages'][0]['content'])
            self.assertIn('Question: Question?',body['messages'][1]['content'])
            return httpx.Response(200,json={'id':'mock','object':'chat.completion','created':1,
                'model':'openai/gpt-5-nano','choices':[{'index':0,'finish_reason':'stop',
                'message':{'role':'assistant','content':'Answer [1]'}}]})
        with patch.object(provider,'client',return_value=self.transport(handler)):
            self.assertEqual(provider.answer('Question?','[1] Source'),('Answer [1]','openai/gpt-5-nano'))

    def test_provider_errors_are_safe(self):
        for status,expected in [(401,'API key'),(402,'credits'),(429,'rate limit')]:
            with self.subTest(status=status):
                def handler(request):
                    return httpx.Response(status,json={'error':{'message':'test-secret'}})
                with patch.object(provider,'client',return_value=self.transport(handler)):
                    with self.assertRaises(HTTPException) as result:
                        provider.embed(['test'])
                self.assertIn(expected,result.exception.detail)
                self.assertNotIn('test-secret',result.exception.detail)


if __name__ == '__main__':
    unittest.main()
