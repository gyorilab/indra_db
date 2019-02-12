import sys
from setuptools import setup, find_packages

def main():
    setup(name='indra_db',
          version='0.0.1',
          description='INDRA Database',
          long_description='INDRA Database',
          url='https://github.com/indralab/indra_db',
          packages=find_packages(),
          install_requires=['boto3', 'sqlalchemy', 'psycopg2-binary',
                            'pgcopy', 'matplotlib', 'flask', 'nltk',
                            'reportlab']
          )


if __name__ == '__main__':
    main()
