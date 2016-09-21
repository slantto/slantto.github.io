#!/usr/bin/env python

import sys

for line in sys.stdin.readlines():

	line=line.split('G')
	print line[0]
