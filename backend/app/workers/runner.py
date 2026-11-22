import sys
import asyncio
import logging
import argparse

from app.workers.stt_worker import run_stt_worker
from app.workers.vision_worker import run_vision_worker
from app.workers.scoring_worker import run_scoring_worker
from app.db.postgres import init_db
from app.db.redis_bus import redis_bus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("oculus.workers.runner")


async def main():
    parser = argparse.ArgumentParser(description="OCULUS Async Worker Mesh Runner")
    parser.add_argument("--worker", choices=["stt", "vision", "scoring", "all"], default="all")
    args = parser.parse_args()

    # Initialize DB & Redis connection
    await init_db()
    await redis_bus.connect()

    stop_event = asyncio.Event()

    tasks = []
    if args.worker in ["stt", "all"]:
        tasks.append(asyncio.create_task(run_stt_worker(stop_event)))
    if args.worker in ["vision", "all"]:
        tasks.append(asyncio.create_task(run_vision_worker(stop_event)))
    if args.worker in ["scoring", "all"]:
        tasks.append(asyncio.create_task(run_scoring_worker(stop_event)))

    logger.info(f"Worker runner active for targets: {args.worker}")
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Stopping workers...")
        stop_event.set()
        await redis_bus.close()


if __name__ == "__main__":
    asyncio.run(main())
