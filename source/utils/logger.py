import logging


class Tee(object):
    """stdout/stderr을 log 파일 + console 양쪽으로 동시에 출력"""
    def __init__(self, name, mode, stream):
        self.file = open(name, mode, encoding='utf-8')
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        
        
def setup_main_logger(log_path):
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_path, encoding='utf-8')
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    return logger
