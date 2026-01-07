from setuptools import setup, find_packages


def main():
    packages = find_packages()
    print("Installing `indra_db` Packages:\n", '\n'.join(packages))
    extras_require = {'test': ['nose', 'coverage', 'python-coveralls',
                               'nose-timer'],
                      'service': ['flask', 'flask-jwt-extended', 'flask-cors',
                                  'flask-compress', 'numpy'],
                      'cli': ['click', 'boto3'],
                      'copy': ['pgcopy'],
                      'misc': ['matplotlib', 'numpy']}
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
          install_requires=['sqlalchemy<2', 'psycopg2-binary', 'cachetools',
                            'termcolor', 'bs4', 'pyyaml'],
          extras_require=extras_require,
          entry_points="""
          [console_scripts]
          indra-db=indra_db.cli:main
          indra-db-service=indra_db_service.cli:main
          indra-db-benchmarker=benchmarker.cli:main
          """)


if __name__ == '__main__':
    main()
