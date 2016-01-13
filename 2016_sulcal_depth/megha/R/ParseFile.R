ParseFile <- function(filename, delimiter="\t", header=FALSE){
# This function parses a plain text file with specified delimiter with no missing values.
	
	ID <- read.table(pipe(paste("cut -f2", filename)), header=header, sep=delimiter, comment.char="")   # read the second column (subject ID)
	SubjID = ID[[1]]   # extract subject IDs
	Nsubj <- length(SubjID)   # calculate the number of subjects
	
	line <- read.table(pipe(paste("cut -f3-", filename)), header=header, sep=delimiter, nrow=1, comment.char="")   # read the first line of the file from the third column
	Ncol = ncol(line)   # calculate the number of columns for quantitative data
	
	con <- pipe(paste("cut -f3-", filename))   # read the file from the third column
	Data <- matrix(scan(con, what=numeric(), skip=as.integer(header), sep=delimiter, nlines=Nsubj, n=Nsubj*Ncol, comment.char=""), Nsubj, Ncol, byrow=TRUE)   # extract quantitative data
	close(con)   # close the file
	
	if (header){
		PhenoNames <- as.character(t(read.table(pipe(paste("cut -f3-", filename)), header=FALSE, sep=delimiter, nrow=1, comment.char="")))   # get the names of the phenotypes
	} else{
		PhenoNames <- "NA"   # no headerline
	}
	
	return(list(Nsubj=Nsubj, Ncol=Ncol, SubjID=SubjID, Data=Data, PhenoNames=PhenoNames))
}