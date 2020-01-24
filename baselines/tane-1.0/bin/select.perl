#!/usr/bin/perl -w
#
# select.perl
# 
# This file is part of TANE version 1.0.
# This is Copyrighted Material. See COPYING file for copying permission.
# Any comments or suggestions; see latest information at
# http://www.cs.helsinki.fi/research/fdk/datamining/tane.
#  

die "usage: $0 <description file>\n" unless (@ARGV == 1);
$DESCRIPTION = "$ARGV[0]";
($path,$base) = $DESCRIPTION =~ /(.*\/|)(.*?)(\..*|$)/;

#######
# types
#######
%off = ( "OFF" => 1, "NO" => 1, "FALSE" => 1 );
%type = ( 
	"TOINTEGERS" => "boolean",
#	"REMOVEDUPLICATES" => "boolean",
	"DATAIN" => "file",
	"ATTRIBUTESIN" => "file",
	"ATTRIBUTESOUT" => "file",
	"STANDARDOUT" => "file",
#	"TANEOUT" => "file",
	"SAVNIKFLACHOUT" => "file",
);

##########
# defaults
##########
$val{TOINTEGERS} = "ON";
#$val{REMOVEDUPLICATES} = "ON";
$val{BASENAME} = $base;
$val{NOOFCOPIES} = 1;

##################
# read description
##################
open DESCRIPTION or die "Can't open description file $DESCRIPTION: $!\n";
while (<DESCRIPTION>) {
	next if /^#/;
	next unless /=/;
	chomp;
	my ($name, $value) = /^\s*(.*?)\s*=\s*(.*?)\s*$/;
	$value =~ s/\$BASENAME/$val{BASENAME}/eg;
	if (defined $type{uc($name)}) {
		if ($type{uc($name)} eq "boolean") {
			$value = "" if $off{uc($value)};
		} elsif ($type{uc($name)} eq "file") {
			$value = $path.$value unless $value =~ m#^(/|-$)#;
		}
	}	
#	print uc($name), "=$value\n";
	$val{uc($name)} = $value;
}

###########
# set umask
###########
if ($val{UMASK}) {
	umask oct($val{UMASK});
}

######################
# read attribute names
######################
@attr_names = ();
if ($val{ATTRIBUTESIN}) {
	open ATTRIBUTESIN, $val{ATTRIBUTESIN} or
				die "Can't open file $val{ATTRIBUTESIN}: $!\n";
	while (<ATTRIBUTESIN>) { chomp; push(@attr_names, $_) unless /^\s*$/; }
} else {
	open DATAIN, $val{DATAIN} or
				die "Can't open file $val{DATAIN}: $!\n";
	my @firstline = split ',', <DATAIN>;
	close DATAIN;
	@attr_names = map index_to_attribute($_), 0..$#firstline;
}

###################
# select attributes
###################
@attributes = ();
if ($val{ATTRIBUTES}) {
	@attributes = split //, $val{ATTRIBUTES};
	@attributes = map attribute_to_index($_), @attributes;
	($max_attribute) = sort {$b <=> $a} @attributes;
	if ($max_attribute > $#attr_names) {
		die "Data does not have attribute ", 
					index_to_attribute($max_attribute), "\n";
	}
	@attr_names = @attr_names[@attributes];
}
$nAttributes = @attr_names;

#######################
# write attribute names
#######################
if ($val{ATTRIBUTESOUT}) {
	open ATTRIBUTESOUT, ">".$val{ATTRIBUTESOUT} or
				die "Can't open file $val{ATTRIBUTESOUT} for output: $!\n";
	local $" = "\n";
	print ATTRIBUTESOUT "@attr_names\n";
}

##############
# select lines
##############
@lines = ();
if ($val{LINES}) {
	@lines = split(",", $val{LINES});
	@lines = map /-/ ? ($`..$') : $_, @lines;
	@lines = sort {$a <=> $b} @lines;
#	print "lines = @lines\n";
	if ($lines[0] < 1) { die "Line number must be 1 or greater.\n"; }
}

####################################
# prepare for standard format output
####################################
if ($val{STANDARDOUT}) {
	$val{STANDARDOUT} = $path.$val{STANDARDOUT} 
					unless $val{STANDARDOUT} =~ m#^[/-]#;
	open STANDARDOUT, ">".$val{STANDARDOUT} or
			die "Can't open ", $val{STANDARDOUT}, " for writing: $!\n";
}

########################################
# prepare for Savnik&Flach format output
########################################
if ($val{SAVNIKFLACHOUT}) {
	$val{SAVNIKFLACHOUT} = $path.$val{SAVNIKFLACHOUT} 
					unless $val{SAVNIKFLACHOUT} =~ m#^[/-]#;
	open SAVNIKFLACHOUT, ">".$val{SAVNIKFLACHOUT} or
				die "Can't open ", $val{SAVNIKFLACHOUT}, " for writing: $!\n";
	my $attrs = join(",", map {index_to_attribute($_)} 0..$nAttributes-1);
	print SAVNIKFLACHOUT "relation( $val{BASENAME}, [$attrs] ).\n";
}



#==============================================
# MAIN LOOP: read and process data line by line
#==============================================
die "Data file not given in $DESCRIPTION\n" unless defined $val{DATAIN};
open(DATA, $val{DATAIN}) or die "Can't open data file $val{DATAIN}: $!\n";
$nLines = 0;
while (<DATA>) {
	chomp;
	my @tuple = split ",";
	last if @tuple == 0;     # stop at empty line
#	print "@tuple\n";
	$nLines++;
	my $nRepeats = 1;
##############
# select lines
##############
	if ($val{LINES}) {
		last if @lines == 0;
		next if $nLines != $lines[0];
		shift @lines;
		while (@lines and $nLines == $lines[0]) {
			$nRepeats++;
			shift @lines;
		}
	}
#	print "@tuple\n";
###################
# select attributes
###################
	if (@tuple < $nAttributes) {
		die "Not enough attributes on line ", 
						$nLines, " of $val{DATAIN}\n";
	}
	if ($val{ATTRIBUTES}) {
		@tuple = @tuple[@attributes];
	}
#	print "@tuple\n";
############
# integerize
############
	if ($val{TOINTEGERS}) {
#		next if $processed{$tuple};
#		$processed{$tuple} = 1;
		my $i;
		for ($i=0; $i<@tuple; $i++) {
			if (not defined $values[$i]{$tuple[$i]}) {
				$values[$i]{$tuple[$i]} = keys(%{$values[$i]});
			}
			$tuple[$i] = $values[$i]{$tuple[$i]};
		}
	}
#	print "@tuple\n";
#############
# make copies
#############
	my $i;
	for ($i=0; $i<$val{NOOFCOPIES}; $i++) {
#		print "copy $i\n";
		my @copy;
		if ($val{TOINTEGERS}) {
			@copy = map { $_*$val{NOOFCOPIES}+$i+1 } @tuple;
		} elsif ($i > 0) {
			@copy = map { "$_§$i" } @tuple;
		} else {
			@copy = @tuple;
		}
#		print "copy $i: @copy\n";
		for (1..$nRepeats) {
#			print "repeat $_\n";
			my $tuple = join(',', @copy);
			if ($val{STANDARDOUT}) { 
				print STANDARDOUT "$tuple\n"; 
			}
			if ($val{SAVNIKFLACHOUT}) {
				print SAVNIKFLACHOUT "$val{BASENAME}($tuple).\n";
			}
		}
	}
}




#########################
# map A-Za-z0-9+/ to 0-63
#########################
sub attribute_to_index {
	my $attr = shift;
	unless ($attr =~ tr#A-Za-z0-9_+##) { 
		die "Attribute \'$attr\' not in [A-Za-z0-9_+]\n"; 
	}
	$attr =~ tr#A-Za-z0-9_+# -_#;
	my $index = ord($attr)-ord(' ');
	if ($index < 0 or $index > 63) { die "Attribute not in [A-Za-z0-9_+]\n"; }
	return $index;
}
#########################
# map 0-63 to A-Za-z0-9+/
#########################
sub index_to_attribute {
	my $index = shift;
	if ($index < 0 or $index > 63) { die "Illegal attribute index $index\n"; }
	my $attr = chr($index+ord(' '));
	$attr =~ tr# -_#A-Za-z0-9_+#;
	return $attr;
}


