#  prerequisite.R
#  
#  Copyright 2013 Vincent FROUIN <vf140245@is207857>
#  
#  June 12th 2013

# If not installed do : (version R >=3  and version BioConductor >= 2.12)
#under the R prompt

# setup repos info
source("http://bioconductor.org/biocLite.R")
#general
biocLite()
# previous snpMatrix package
biocLite("snpStats")
#domain specific info
biocLite("GO.db")
biocLite("org.Hs.eg.db")
biocLite("GSEABase")
biocLite("lumi")
biocLite("lumiHumanAll.db")
biocLite("lumiHumanIDMapping")
biocLite("KEGG.db")
biocLite("KEGGgraph")