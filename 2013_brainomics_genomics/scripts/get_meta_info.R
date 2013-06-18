#  get_meta_info.R
#  
#  Copyright 2013 Vincent FROUIN <vf140245@is207857>
#  
#  June 12th 2013

library(GSEABase)
library(GO.db)
library(org.Hs.eg.db)
library(KEGG.db)
library(RJSONIO)

# example with GO #############################################################
# Build meta from GO db
# Query on GO db
# 1) get
goterm <- as.list(GOTERM)
goterm.term = lapply(goterm, Term)
# 2) example for two terms
pathw.synaptic = goterm.term[grep("synaptic", goterm.term)]
#pathw.apolipo_receptor = goterm.term[grep("apolipoprotein receptor", goterm.term)]


# 3) Query on the db  : eg(entrezgene) <-> GO
# Map geneId to geneName
eg2s <- org.Hs.egSYMBOL
mapped_genes <- mappedkeys(eg2s)
eg2s <- as.list(eg2s[mapped_genes])
# Get the gene symbol that are mapped to an entrez gene identifiers
gene2go = toTable(org.Hs.egGO)
gene2go = data.frame(gene2go$go_id, gene2go$Evidence, gene2go$gene_id)
gene2go.goframe   = GOFrame(gene2go, organism = "Homo sapiens")
gene2go.goAllFrame = GOAllFrame(gene2go.goframe)


# 4) use geneSetCollection object from package GSEA 
# 4.1) build sets of gens per pathway
geneSetCollection <- GeneSetCollection(gene2go.goAllFrame, setType = GOCollection())
# 4.2) Query in the list
pathw.synaptic.GSCgeneId = lapply(intersect(names(pathw.synaptic), names(geneSetCollection)), 
                      function(s) geneSetCollection[[s]])

pathw.synaptic.GSCgeneName <- mapply(function(geneIds, goId) {
                gs = as.vector(sapply(geneIds,function(s) eg2s[[s]]))
                #gs = intersect(gs, gene.info[,1])
                if (length(gs))
                  GeneSet(gs,
                          geneIdType=SymbolIdentifier("org.Hs.eg.db"),
                          collectionType=GOCollection(goId),
                          setName=goterm.term[[goId]])
                                      },
               lapply(pathw.synaptic.GSCgeneId,geneIds), lapply(pathw.synaptic.GSCgeneId,setName))
# 5) House keeping 
pathw.synaptic.GSCgeneName <- pathw.synaptic.GSCgeneName[which(
                                          sapply(pathw.synaptic.GSCgeneName,
                                          function(s) !is.null(s) ))
                                          ]
pathw.synaptic.GSCgeneName <- GeneSetCollection(pathw.synaptic.GSCgeneName)
pathw.synaptic.SETgeneName <- lapply(pathw.synaptic.GSCgeneName, geneIds)
names(pathw.synaptic.SETgeneName) <- lapply(pathw.synaptic.GSCgeneName, setName)


# get snp info
# snp.info <- read.delim("/neurospin/brainomics/2013_brainomics_genomics/data/geno/snp_b37_intersect_maf5hwe4_qcsubject.bim", header=F, stringsAsFactors=FALSE)
# snp.info = data.frame(Snp.name=snp.info[,"V2"],chr=snp.info[,"V1"],pos=snp.info[,"V4"])


# 6) save a jason file
cat(toJSON(pathw.synaptic.SETgeneName),file="/neurospin/brainomics/2013_brainomics_genomics/data/go_synaptic.json")

# example with GO #############################################################
# example with KEGG ###########################################################
path2egid <- as.list(org.Hs.egPATH2EG)
# Remove pathway identifiers that do not map to any entrez gene id
path2egid <- path2egid[!is.na(path2egid)]

eg2symbol <- org.Hs.egSYMBOL
# Get the gene names that are mapped to an entrez gene identifier
mapped_genes <- mappedkeys(eg2symbol)
# Convert to a list
eg2symbol <- as.list(eg2symbol[mapped_genes])
# get id 2 nme link
pathid2name <- as.list(KEGGPATHID2NAME)

# 1) now build the list
kegg.pathway <- sapply(path2egid, function(a){
                      as.vector(sapply(a, function(i) eg2symbol[[i]]))
                      })
# 2) House keeping
nm <- sapply(names(path2egid), function(a){ pathid2name[[a]] })
names(kegg.pathway) <- nm
# 6) save a jason file
cat(toJSON(kegg.pathway),file="/neurospin/brainomics/2013_brainomics_genomics/data/kegg.json")

