#!/usr/bin/python
# a short script to extract from output CG iterations, Total Wall Time for solution and RHS assembly etc.
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
output_file = input_file+'_ndofs_per_atom.parsed'

fin = open(input_file, 'r')
fout = open(output_file, 'w')

pattern = r'[+\-]?(?:[0-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'

fout.write('# Natoms | number of dofs\n')
cycle = -1
natoms = -1
for line in fin:
# If blank line go to next line
    if not line.strip():
	fout.write('\n')
        continue
    else:
        if 'Number of atoms' in line:
            natoms = re.findall(pattern, line)[0]
	    fout.write('{0}'.format(natoms)) 
            #fout.write('\n')    
        if 'Cycle' in line:
            #if cycle >= 0:
             #   fout.write('\n')
            cycle = re.findall(pattern, line)[0]
            #fout.write('\t{0}'.format(cycle))
	if 'Number of degrees of freedom' in line:
	    ndofs = re.findall(pattern, line)[0]
	    line_striped = line.lstrip()
	    count = 0
            # line does not start with a number and we already parsed Cycle line:
            if not line_striped[0].isdigit():
	        # now add all numbers we can find in the line:
                for item in re.findall(pattern, line):
		    count = count + 1
		    #fout.write('{0}'.format(count))
		    if count == 1:
	                fout.write('\t{0}'.format(item))
	    #fout.write('\n')
        
        if 'Starting epilogue' in line:
           fout.close()
           break

print "done!"
