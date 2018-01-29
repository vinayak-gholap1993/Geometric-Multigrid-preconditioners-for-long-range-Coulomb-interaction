#!/usr/bin/python
# a short script to extract from output RHS assembly Wall Time
import argparse
import re


# define command line arguments
parser = argparse.ArgumentParser(description='a short script to extract from '
                                             'output relevant data. '
                                             'This is '
                                             'convinient for plotting')
parser.add_argument('file', metavar='file',
                    help='Output file.')
args = parser.parse_args()

print 'Parse', args.file, 'input file...'
# ready to start...
input_file = args.file
output_file = input_file+'_RHS_time.parsed'

fin = open(input_file, 'r')
fout = open(output_file, 'w')

pattern = r'[+\-]?(?:[0-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'

fout.write('# Natoms | RHS assembly walltime\n')
cycle = -1
natoms = -1
for line in fin:
# If blank line go to next line
    if not line.strip():
	#fout.write('\n')
        continue
    else:
        if 'Number of atoms' in line:
            natoms = re.findall(pattern, line)[0]
	    fout.write('\n')
	    fout.write('{0}'.format(natoms))                
        if 'RHS assembly' in line:
	    if not 'RHS assembly optimization' in line:
      	        ncells = re.findall(pattern, line)[0]
	        #fout.write('\t{0}'.format(ncells))
		line_striped = line.lstrip()
		count = 0
                # line does not start with a number and we already parsed Cycle line:
                if not line_striped[0].isdigit():
                    # now add all numbers we can find in the line:
                    for item in re.findall(pattern, line):
			count = count + 1
			#fout.write('{0}'.format(count))
			if count == 2:
                            fout.write('\t{0}'.format(item))
	    #fout.write('\n')
        
        if 'Starting epilogue' in line:
           fout.close()
           break

print "done!"
