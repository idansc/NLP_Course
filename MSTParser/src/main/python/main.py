import os
import time

from dataparser import Parser

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    config = {
        'training_data': "../resources/train_sample.labeled"
    }
    
    print("Beginning parsing...")
    start_time = time.process_time()
    parser = Parser(config['training_data'])
    print("Done. Elapsed time:", (time.process_time() - start_time) / 60, "secs\n")
    
    
