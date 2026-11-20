import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(base_policy.BasePolicy):
    """Send observations to an EVA policy server over websocket."""

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                connection = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                return connection, msgpack_numpy.unpackb(connection.recv())
            except ConnectionRefusedError:
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
