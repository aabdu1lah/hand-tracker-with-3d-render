import websockets
import json
from typing import Set, Dict, List, Any

class Server:
    """
    Manages WebSocket connections and broadcasts data to connected clients.
    
    This class acts as a simple relay hub: it accepts connections, tracks them,
    and formats data payloads to be sent out to all active listeners (e.g., a frontend UI).
    """
    
    # A thread-safe set to store active client connections.
    # Using a set ensures uniqueness and allows O(1) adds/removes.
    CONNECTED_CLIENTS: Set[websockets.ServerConnection] = set()

    async def register_client(self, websocket: websockets.ServerConnection) -> None:
        """
        Handler for new WebSocket connections.
        
        This method is passed to the `websockets.serve` function. It keeps the 
        connection alive and manages the client registry.

        Args:
            websocket (websockets.ServerConnection): The active connection object.
        """
        # 1. Register
        Server.CONNECTED_CLIENTS.add(websocket)
        print(f"Client connected. Total: {len(Server.CONNECTED_CLIENTS)}")
        
        try:
            # 2. Keep Alive
            # This awaits until the connection is closed (by client or network error).
            # Without this await, the handler would exit immediately and close the socket.
            await websocket.wait_closed()
        finally:
            # 3. Cleanup
            # This block runs regardless of how the connection ends (error or graceful),
            # ensuring we don't try to send data to dead sockets later.
            Server.CONNECTED_CLIENTS.remove(websocket)
            print(f"Client disconnected. Total: {len(Server.CONNECTED_CLIENTS)}")

    def construct_payload(self, hands_contract: List[Dict[str, Any]], timestamp_ms: int) -> str:
        """
        Formats the tracking data into a JSON string suitable for transmission.

        Args:
            hands_contract (List[Dict[str, Any]]): The list of detected hands and their states.
            timestamp_ms (int): The synchronization timestamp.

        Returns:
            str: A JSON-encoded string.
        """
        payload = {
            "timestamp": timestamp_ms,
            "hands": hands_contract
        }
        return json.dumps(payload)

    def serve(self, *args, **kwargs) -> Any:
        """
        Wrapper around `websockets.serve`.
        
        Returns an async context manager that runs the server loop.
        
        Usage:
            async with server.serve(handler, host, port):
                await asyncio.Future() # run forever
        """
        return websockets.serve(*args, **kwargs)