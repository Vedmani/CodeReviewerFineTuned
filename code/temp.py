import logging
def main():
    print('printed from main')
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Hello world")
    print("hello world")
    mai()