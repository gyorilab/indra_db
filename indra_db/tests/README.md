# Testing the INDRA Database

In `indra_db`, we use the `nosetests` framework to run tests. Tests are 
automatically detected in the usual ways, such as by the prefix `test_` on
files and functions.

## Setting up the Test Database
Most tests require access to a test database, which is, and should remain,
separate from the database generally used. This repository requires a database
of at least postgers version 9.6, which for most systems will require some
extra work, as 9.6 is not (or at least was not for me) natively available
through `apt-get`.

To get access to the latest versions of postgres, you must first execute the
following (a la [this site](https://r00t4bl3.com/post/how-to-install-postgresql-9-6-on-linux-mint-18-1-serena)):
```bash
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt/ xenial-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
```
And optionally
```bash
sudo apt-get upgrade
```
You should now see there are several versions of postgres available for
installation. You should be able to install any version >= 9.6, but for the
sake of simplicity, I will from here assume 9.6 is being installed.
```bash
sudo apt-get install postgresql-9.6 postgresql-common
```

Also, note that this is all much more complicated if you have or have ever had
a different version of postgres installed. One way to check this is to inspect
the `/etc/postgresql` direcotry for other versions. This will indicate current
active versions, but also version that were uninstalled without `--purge`,
which could still interfere with the running database.

You can also run the `pg_lsclusters` command to see what clusters are currently
running. You should see only one, with the correct version, running on port
5432, like so:
```
Ver Cluster Port Status Owner    Data directory               Log file
9.6 main    5432 online postgres /var/lib/postgresql/9.6/main /var/log/postgresql/postgresql-9.6-main.log
```

Lastly, you should check and make sure that when you `la /var/run/postgresql/`
(note the `la` for list all, not `ls`) you see the following:
```
9.6-main.pg_stat_tmp  9.6-main.pid  .s.PGSQL.5432  .s.PGSQL.5432.lock
```
If you don't see this, you may need to reboot or take other actions.

Once all the above is confirmed, you will need to make access to the database
more permissive. *You should **not** do this when the database could be
exposed to the outside or multiple users may be using the same machine*.

Edit the the host-based authentication (HBA) config file: `pg_hba.conf`, which
will likely require `sudo`. For me, this file is located at 
`/etc/postgresql/9.6/main/pg_hba.conf`. For the sake of this test setup you
should got to the bottom where you see several lines of the form:
```
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             postgres                                peer
```
Changing `peer` or `md5` in the `METHOD` sections to `trust`. Save the file,
and then you will need to reboot your computer for the changes to take effect.

Once that is done, you can create the test database that EMMAA uses:
`emmaadb_test` by entering the following command:
```bash
sudo -u postgres createdb indradb_test
```
You should not be prompted to enter a password. If so, revisit the changes made
to the `pg_hba.conf` file, and again make sure you rebooted after making the
changes. You can then test that the database works as expected by entering
```bash
psql -U postgres
```
At which point you should see a prompt like this:
```
psql (10.9 (Ubuntu 10.9-1.pgdg16.04+1), server 9.6.14)
Type "help" for help.

postgres=# 

```
Enter `\q` to exit the prompt, and you should be all set to run the tests.

