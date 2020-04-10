import sys

import bucket_api
import data_library

def main(argv):
    bucketapi = bucket_api.get_bucket_api()
    bucketapi.upload_file(data_library.CATEGORIES_PATH, './demodata/categories.json')

if __name__ == '__main__':
    main(sys.argv)