from setuptools import setup, find_packages


def main():
    packages = find_packages()
    print("Installing `indra_db` Packages:\n", '\n'.join(packages))
    extras_require = {'test': ['nose', 'coverage', 'python-coveralls',
                               'nose-timer'],
                      'rest_api': ['flask', 'flask-jwt-extended', 'flask-cors',
                                   'flask-compress']}
    extras_require['all'] = list({dep for deps in extras_require.values()
                                  for dep in deps})
    setup(name='indra_db',
          version='0.0.1',
          description='INDRA Database',
          long_description='INDRA Database',
          url='https://github.com/indralab/indra_db',
          author='Patrick Greene',
          author_email='patrick_greene@hms.harvard.edu',
          packages=packages,
          include_package_data=True,
          install_requires=['indra', 'boto3', 'sqlalchemy', 'psycopg2',
                            'pgcopy', 'matplotlib', 'nltk', 'reportlab',
                            'cachetools', 'termcolor'],
          extras_require=extras_require)


if __name__ == '__main__':
    main()
