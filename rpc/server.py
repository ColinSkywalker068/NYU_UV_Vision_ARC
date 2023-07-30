# Python SDK server for the custom base component
import sys
import time
import asyncio
from viam.rpc.server import Server

from xavier_rpc_driver import SentryDriver

# Add another directory where Python will look when importing modules
sys.path.append("/usr/local/lib/python3.11/site-packages")



async def main():
    time.sleep(5)  # Give the sentry server time to start up
    srv = Server([SentryDriver("sentry-base")])
    await srv.serve()


if __name__ == "__main__":
    asyncio.run(main())