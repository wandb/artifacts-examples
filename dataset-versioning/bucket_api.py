import shutil
import hashlib
import os
import sys
import boto3

class BucketApiS3(object):
    def __init(self, bucket_name):
        self._bucket_name = bucket_name
        self._s3 = boto3.client('s3')
    
    def download_file(self, key, local_path, hash=None):
        # TODO: hash implementation: don't download if we already have the correct
        #   has at local_path.
        self._s3.download_file(self._bucket_name, key, local_path)

    def upload_file(self, key, local_path):
        self._s3.upload_file(local_path, self._bucket_name, key)

    def get_hash(self, key):
        head_obj = self._s3.head_object(Bucket=self._bucket_name, Key=key)
        return head_obj['ETag']


class BucketApiLocal(object):
    def __init__(self, local_dir):
        self._local_dir = local_dir
    
    def download_file(self, key, local_path, hash=None):
        dirname = os.path.dirname(local_path)
        os.makedirs(dirname, exist_ok=True)
        try:
            if hash is None or (
                    not os.path.isfile(local_path) or self._file_hash(local_path) != hash):
                shutil.copyfile(os.path.join(self._local_dir, key), local_path)
            return True
        except FileNotFoundError:
            return False

    def upload_file(self, key, local_path):
        file_path = os.path.join(self._local_dir, key)
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)
        shutil.copy(local_path, file_path)

    def _file_hash(self, local_path):
        hash_md5 = hashlib.md5()
        with open(local_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_hash(self, key):
        return self._file_hash(os.path.join(self._local_dir, key))

def get_bucket_api():
    bucket_settings_fname = 'bucket_settings.txt'
    if not os.path.isfile(bucket_settings_fname):
        print('Missing %s, please see README.md' % bucket_settings_fname)
        sys.exit(1)
    bucket_url = open(bucket_settings_fname).read().strip()
    if bucket_url.startswith('local://'):
        local_path = bucket_url.lstrip('local://')
        return BucketApiLocal(local_path)
    elif bucket_url.startswith('s3://'):
        bucket_name = bucket_url.lstrip('s3://')
        return BucketApiLocal(bucket_name)
    print('Invalid bucket url. Please see README.md. %s' % bucket_url)
    sys.exit(1)
