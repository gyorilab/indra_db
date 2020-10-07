import json

import boto3
import logging
from os import path
from flask import Flask, jsonify

from benchmarker.util import list_stacks, list_apis

logger = logging.getLogger('benchmark_viewer')

HERE = path.dirname(__file__)

app = Flask('benchmark_viewer')
BUCKET = 'bigmech'
PREFIX = 'indra-db/benchmarks/'


def load(**kwargs):
    with open(path.join(HERE, 'benchmark.html'), 'r') as f:
        s = f.read()
    for key, value in kwargs.items():
        s = s.replace(f'{{{{ {key} }}}}', json.dumps(value))
    return s


@app.route('/', methods=['GET'])
def serve_page():
    return load(stacks=list_stacks(), apis=list_apis())


@app.route('/fetch/<corpus_name>/<stack_name>/<test_file>', methods=['GET'])
def get_stack_data(corpus_name, stack_name, test_file):
    try:
        s3 = boto3.client('s3')
        file = s3.get_object(
            Bucket=BUCKET,
            Key=f'{PREFIX}{corpus_name}/{stack_name}/{test_file}'
        )
        data = json.loads(file['Body'].read())
    except Exception as e:
        logger.exception(e)
        return jsonify({'message': f'Error: {e}'}), 500
    return jsonify({'message': 'success', 'tests': data}), 200


@app.route('/list/<corpus_name>', methods=['GET'])
def list_corpus_options(corpus_name):
    option_dict = {}
    try:
        s3 = boto3.client('s3')
        prefix = f'{PREFIX}{corpus_name}/'
        res = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        keys = [e['Key'][len(prefix):] for e in res['Contents']]
        for key in keys:
            stack, test = key.split('/')
            test_time = test.split('.')[0]
            label = f'{test_time} ({stack})'
            option_dict[label] = {'stack': stack, 'test': test}
    except Exception as e:
        logger.exception(e)
        return jsonify({'message': f'Error: {e}'}), 500
    return jsonify({'message': 'success', 'options': option_dict})
