# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:03:29 2013

@author: jl237561
"""

import os
from soma_workflow.client import Job, Workflow, Helper

if __name__ == "__main__":
    # define input parameters
    NUM_CHUNK_SNP = 200
    NUM_CHUNK_IMG = 200
    # for test, you can comment them if you want
    NUM_CHUNK_SNP = 2
    NUM_CHUNK_IMG = 2
    num_perms = 32
    # define paths
    cccworkdir = "/ccc/work/cont003/dsv/lijpeng"
    data_path = os.path.join(cccworkdir, "data")
    filename_python = "%s/brainomics/git/brainomics/ml/mulm_gpu"\
                      "/mulm/bench/example_data.py" % cccworkdir
    filename_cov_chunk = "%s/cov.npy" % data_path
    # define jobs
    jobs = []
    for i_snp_chunk_index in xrange(NUM_CHUNK_SNP):
        for i_img_chunk_index in xrange(NUM_CHUNK_IMG):
            filename_snp_chunk = "%s/snp_chunk_%d.npz" % \
                                 (data_path, i_snp_chunk_index)
            filename_image_chunk = "%s/tr_image_chunk_%d.npz" % \
                                   (data_path, i_img_chunk_index)
            # create the workflow:
            job = Job(command=["python",
                                 filename_python,
                                 filename_snp_chunk,
                                 filename_image_chunk,
                                 filename_cov_chunk,
                                 "32"
                                 ], name="snp=%d, img=%d" % \
                                 (i_snp_chunk_index, i_img_chunk_index))
            jobs.append(job)
    dependencies = []
    workflow = Workflow(jobs=jobs,
                        dependencies=dependencies)
    # save the workflow into a file
    Helper.serialize("./export_chunks_soma_workflow.wf", workflow)