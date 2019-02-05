#!/bin/bash

sudo -u postgres psql <<PGSCRIPT
CREATE database profiler;
CREATE user profileruser;
ALTER USER profileruser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES on database profiler to profileruser ;
PGSCRIPT

echo "PG database and user has been created."
