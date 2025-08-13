"""
Main entry point for the text renderer application.

This module provides the command-line interface and multiprocessing infrastructure
for generating synthetic text images using the text_renderer library.
"""

import argparse
import multiprocessing as mp
import os
import time
from multiprocessing.context import Process

import cv2
from loguru import logger

from text_renderer.config import GeneratorCfg, get_cfg
from text_renderer.dataset import ImgDataset, LmdbDataset
from text_renderer.render import Render

cv2.setNumThreads(1)

STOP_TOKEN = "kill"

# each child process will initialize Render in process_setup
render: Render


class DBWriterProcess(Process):
    """
    Database writer process for handling dataset storage operations.

    This process runs in a separate process and handles writing generated images
    and labels to the dataset storage (either LMDB or image files). It provides
    progress logging and error handling for the storage operations.

    Args:
        dataset_cls: Dataset class to use for storage (LmdbDataset or ImgDataset)
        data_queue: Multiprocessing queue for receiving image data
        generator_cfg (GeneratorCfg): Configuration for the generation process
        log_period (float): Logging period as percentage of total images (default: 1)
    """

    def __init__(
        self,
        dataset_cls,
        data_queue,
        generator_cfg: GeneratorCfg,
        log_period: float = 1,
    ):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.data_queue = data_queue
        self.generator_cfg = generator_cfg
        self.log_period = log_period

    def run(self):
        """
        Main process loop for handling dataset storage operations.

        This method continuously reads from the data queue and writes images
        to the dataset until it receives a stop token. It provides progress
        logging and handles the complete storage pipeline.
        """
        num_image = self.generator_cfg.num_image
        save_dir = self.generator_cfg.save_dir
        log_period = max(1, int(self.log_period / 100 * num_image))
        try:
            with self.dataset_cls(str(save_dir)) as db:
                exist_count = db.read_count()
                count = 0
                logger.info(f"Exist image count in {save_dir}: {exist_count}")
                start = time.time()
                while True:
                    m = self.data_queue.get()
                    if m == STOP_TOKEN:
                        logger.info("DBWriterProcess receive stop token")
                        break

                    name = "{:09d}".format(exist_count + count)
                    db.write(name, m["image"], m["label"])
                    count += 1
                    if count % log_period == 0:
                        logger.info(
                            f"{(count/num_image)*100:.2f}%({count}/{num_image}) {log_period/(time.time() - start + 1e-8):.1f} img/s"
                        )
                        start = time.time()
                db.write_count(count + exist_count)
                logger.info(f"{(count / num_image) * 100:.2f}%({count}/{num_image})")
                logger.info(f"Finish generate: {count}. Total: {exist_count+count}")
        except Exception as e:
            logger.exception("DBWriterProcess error")
            raise e


def generate_img(data_queue):
    """
    Generate a single image and put it in the data queue.

    This function is called by worker processes to generate images
    using the global render instance and put the results in the queue.

    Args:
        data_queue: Multiprocessing queue for sending image data
    """
    data = render()
    if data is not None:
        data_queue.put({"image": data[0], "label": data[1]})


def process_setup(*args):
    """
    Initialize the render instance for a worker process.

    This function is called by each worker process to set up its own
    render instance with a unique random seed.

    Args:
        *args: Arguments passed to the process, first argument should be RenderCfg
    """
    global render
    import numpy as np

    # Make sure different process has different random seed
    np.random.seed()

    render = Render(args[0])
    logger.info(f"Finish setup image generate process: {os.getpid()}")


def parse_args():
    """
    Parse command line arguments for the text renderer application.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="python file path")
    parser.add_argument("--dataset", default="img", choices=["lmdb", "img"])
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--log_period", type=float, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    data_queue = manager.Queue()
    args = parse_args()

    dataset_cls = LmdbDataset if args.dataset == "lmdb" else ImgDataset

    generator_cfgs = get_cfg(args.config)

    for generator_cfg in generator_cfgs:
        db_writer_process = DBWriterProcess(
            dataset_cls, data_queue, generator_cfg, args.log_period
        )
        db_writer_process.start()

        if args.num_processes == 0:
            process_setup(generator_cfg.render_cfg)
            for _ in range(generator_cfg.num_image):
                generate_img(data_queue)
            data_queue.put(STOP_TOKEN)
            db_writer_process.join()
        else:
            with mp.Pool(
                processes=args.num_processes,
                initializer=process_setup,
                initargs=(generator_cfg.render_cfg,),
            ) as pool:
                for _ in range(generator_cfg.num_image):
                    pool.apply_async(generate_img, args=(data_queue,))

                pool.close()
                pool.join()

            data_queue.put(STOP_TOKEN)
            db_writer_process.join()
