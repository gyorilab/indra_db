import json
import boto3
import logging
from os import path
from flask import Flask, jsonify

logger = logging.getLogger('benchmark_viewer')

HERE = path.dirname(__file__)

app = Flask('benchmark_viewer')
BUCKET = 'bigmech'
BASE = 'indra-db/benchmarks'


@app.route('/', methods=['GET'])
def serve_page():
    with open(path.join(HERE, 'benchmark.html'), 'r') as f:
        return f.read().replace('{{ stacks }}', json.dumps(['spruce', 'pine']))


@app.route('/fetch/<stack_name>', methods=['GET'])
def get_stack_data(stack_name):
    try:
        s3 = boto3.client('s3')
        res = s3.list_objects_v2(Bucket=BUCKET, Prefix=f'{BASE}/{stack_name}/')
        keys = [e['Key'] for e in res['Contents']]
        result = {}
        for key in keys:
            date_str = path.basename(key).split('.')[0]
            file = s3.get_object(Bucket=BUCKET, Key=key)
            result[date_str] = json.loads(file['Body'].read())
    except Exception as e:
        logger.exception(e)
        return jsonify({'message': f'Error: {e}'}), 500
    return jsonify({'message': 'success', 'tests': result}), 200
