# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Async utilities for concurrent LLM generation.

This module provides utilities for running async generation with LLM servers:
- ChatCompletionProxy: Async interface for submitting chat completion requests
- run_coroutine_in_chat_scheduler_loop: Helper for running coroutines in async threads

Key Classes:
    - ChatCompletionProxy: Async proxy for OpenAI-compatible chat completion endpoints.

Key Functions:
    - run_coroutine_in_chat_scheduler_loop: Execute coroutine in async scheduler thread.

Example:
    >>> proxy = ChatCompletionProxy(config, model_path, server_addresses)
    >>> completion = await proxy.get_chat_completions(
    ...     model="qwen-vl",
    ...     messages=[{"role": "user", "content": "Hello"}]
    ... )
"""

from __future__ import annotations

import asyncio
import heapq
from typing import Any, Callable, Coroutine
from uuid import uuid4

import aiohttp
from httpx import AsyncClient, Limits, Timeout
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion


class ChatCompletionProxy:
    """Async proxy for chat completion requests to LLM servers.

    Provides an async interface for submitting chat completion requests to
    OpenAI-compatible LLM servers. Manages load balancing across multiple
    server addresses and caches request/address mappings.

    Instead of using callback chains, returns ChatCompletion directly, allowing
    natural async/await usage in agent rollout loops.

    Attributes:
        addr_client_map (dict): Mapping of server addresses to HTTP clients.
        session (aiohttp.ClientSession): Shared HTTP session for requests.
        model_name (str): Default model name for requests.
    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: list[str],
        max_cache_size: int = 10000,
    ):
        self.addr_client_map = {}
        conn = aiohttp.TCPConnector(
            limit=len(server_addresses) * 1024,
            limit_per_host=1024,
            keepalive_timeout=600,
            loop=asyncio.get_event_loop(),
        )  # aiohttp use get_running_loop(), but the loop is not launched yet
        self.session = aiohttp.ClientSession(connector=conn)
        super().__init__(config, model_path, server_addresses, max_cache_size)

    def get_client(self, address: str) -> AsyncClient:
        """Get or create HTTP client for a server address.

        Args:
            address (str): Server address (host:port).

        Returns:
            AsyncClient: HTTP client with timeout and connection limits configured.
        """
        return self.addr_client_map.get(
            address,
            AsyncClient(
                timeout=Timeout(connect=60, read=None, write=None, pool=None),
                limits=Limits(max_connections=8192, max_keepalive_connections=8192, keepalive_expiry=600),
            ),
        )

    async def submit_chat_completions(
        self,
        callback: Callable[[ChatCompletion, dict[str, Any], Exception], None],
        callback_additional_info: dict[str, Any],
        **chat_complete_request,
    ):
        """Not implemented in async proxy.

        Raises:
            NotImplementedError: Callback-based submission not supported.
        """
        raise NotImplementedError("ChatCompletionProxy does not support submit_chat_completions")

    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        """Submit request using OpenAI AsyncOpenAI client.

        Args:
            address (str): Server address.
            **chat_complete_request: Chat completion parameters.

        Returns:
            ChatCompletion: Response from server.
        """
        client = AsyncOpenAI(
            base_url=f"http://{address}/v1", api_key="token-abc123", http_client=self.get_client(address)
        )
        return await client.chat.completions.create(**chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        """Submit request using aiohttp directly.

        Args:
            address (str): Server address.
            **chat_complete_request: Chat completion parameters (must include extra_headers).

        Returns:
            ChatCompletion: Response from server.
        """
        extra_headers = chat_complete_request.pop("extra_headers")
        async with self.session.post(
            url=f"http://{address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},
            json=chat_complete_request,
        ) as resp:
            data = await resp.json()
            return ChatCompletion(**data)

    async def get_chat_completions(
        self,
        model: str | None = None,
        **chat_complete_request,
    ) -> tuple[ChatCompletion | None, Exception | None]:
        """Submit a chat completion request with load balancing.

        Routes the request to the server with the least number of pending requests.
        Supports request tracking via x-request-id header.

        Args:
            model (str | None): Model name (uses default if None).
            **chat_complete_request: Chat completion parameters following OpenAI API.
                See https://platform.openai.com/docs/api-reference/chat/create

        Returns:
            tuple[ChatCompletion | None, Exception | None]: (completion, exception).
                If request succeeds, exception is None. If fails, completion is None.
        """
        model = model or self.model_name
        if "extra_headers" not in chat_complete_request:
            chat_complete_request["extra_headers"] = {}

        extra_headers = chat_complete_request["extra_headers"]
        request_id = extra_headers.get("x-request-id", None)
        if request_id:
            if request_id.startswith("chatcmpl-"):
                request_id = request_id[len("chatcmpl-") :]
                extra_headers["x-request-id"] = request_id

            address = self.request_id_to_address[request_id]
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

            request_id = uuid4().hex
            self.request_id_to_address[request_id] = address
            chat_complete_request["extra_headers"]["x-request-id"] = request_id

        completions, exception = None, None
        try:
            # TODO: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(address, model=model, **chat_complete_request)
        except Exception as e:
            # Let user handle the exception
            exception = e

        return completions, exception


def run_coroutine_in_chat_scheduler_loop(async_server, coro: Coroutine) -> Any:
    """Run a coroutine in the async server's chat scheduler event loop.

    Bridges sync and async code by submitting a coroutine to run in a separate
    thread's event loop. Originally designed for chat scheduler methods, now used
    to run AsyncRAgent.rollout() and gather results.

    Args:
        async_server: Async server with chat_scheduler_loop attribute.
        coro (Coroutine): Coroutine to execute in the scheduler loop.

    Returns:
        Any: Result of the coroutine.

    Raises:
        AssertionError: If chat_scheduler is not initialized.
    """
    assert async_server.chat_scheduler is not None, "chat scheduler is not initialized."
    future = asyncio.run_coroutine_threadsafe(coro, async_server.chat_scheduler_loop)
    return future.result()
