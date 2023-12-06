import hashlib
import os.path


def is_valid_checksum(save_path, checksum_path):
    cache_path = save_path + '.SHA256'
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as file:
            found = file.read()
    else:
        with open(save_path, 'rb') as file:
            found = hashlib.sha256(file.read()).hexdigest()
        found = f'{found}  {os.path.basename(save_path)}'
        with open(cache_path, 'w') as file:
            file.write(found)

    with open(checksum_path, 'r') as file:
        expected = file.read()
    is_valid = found == expected
    return is_valid


def test():
    save_path = '/mnt/pool1/binance-public-data/data/spot/monthly/klines/BNBBTC/1m/BNBBTC-1m-2017-07.zip'
    checksum_path = save_path + '.CHECKSUM'
    is_valid_checksum(save_path, checksum_path)


if __name__ == '__main__':
    test()
