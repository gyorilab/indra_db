from setuptools import setup, find_packages


def main():
    packages = find_packages()
    print("Installing `indra_db` Packages:\n", '\n'.join(packages))
    setup(name='indra_db',
          version='0.0.1',
          description='INDRA Database',
          long_description='INDRA Database',
          url='https://github.com/indralab/indra_db',
          author='Patrick Greene',
          author_email='patrick_greene@hms.harvard.edu',
          packages=packages,
          include_package_data=True,
          install_requires=['indra', 'boto3', 'sqlalchemy', 'psycopg2-binary',
                            'pgcopy', 'matplotlib', 'flask', 'nltk',
                            'reportlab', 'cachetools'],
          extras_require={'test': ['nose', 'coverage', 'python-coveralls',
                                   'nose-timer']},
          )


if __name__ == '__main__':
    main()
