
suffix = ".out"

nodecounts = ["1", "2", "4", "8", "16", "24", "32"]
nodecountsint = [1, 2, 4, 8, 16, 24, 32]
nodecountlabels = ["1", "2", "4", "8", "16", "24", "32"]

# nodecounts = ["1", "2", "4", "8", "16", "32"]
# nodecountsint = [1, 2, 4, 8, 16,  32]
# nodecountlabels = ["1", "2", "4", "8", "16", "32"]

xticks = nodecounts

#basepaths = ["sc3-invasion-small-intervals"]
#static_job_name = "inv-static"
#jobtypes = ["inv-static", "inv-global-random", "inv-global-busy",
#            "inv-local-random", "inv-local-busy"]
# yyy = 62

#basepaths = ["sc3-lazyactivation-intervals"]
#static_job_name = "static"
#jobtypes = ["static", "global-random", "global-busy",
#            "local-random", "local-busy"]
# yyy = 62

basepaths = ["sc3-lazyactivation"]
static_job_name = "static"
jobtypes = ["static", "global-random", "global-busy",
            "contig-local-random", "contig-local-busy", "offload-local-single"]
yyy = 52

#basepaths = ["sc3-nodeslowdown"]
#static_job_name = "sw-static"
#jobtypes = ["sw-static", "sw-global-random", "sw-global-busy",
#            "sw-local-random", "sw-local-busy", "sw-offload-local-single"]
#yyy = 32

defined_colors = ["purple", "orangered", "yellowgreen",
                  "orange", "mediumturquoise", "palevioletred", "lightsteelblue", "thistle",
                  "skyblue", "lightsalmon"]

markers = [".", "o", "^", "*", "+", "x", "D", "|"]

linestyles = ["-.", "--", "-"]

workload_model_to_name = {
    "23": "Task Count", "24": "Actor Execution Time as Cost", "27": "Task Count weighted by Actor Cost"}

job_to_label = {"sw-static": "Static", "sw-local-busy": "Local-busy", "sw-offload-local-single": "Offloading",
                "sw-global-busy": "Global-busy", "sw-global-random": "Global-random", "sw-local-random": "Local-random",  "local-random": "Local-random",
                "static": "Static", "local-busy": "Local-busy", "offload-local-single": "Offloading",
                "global-busy": "Global-busy", "global-random": "Global-random", "local-random": "Local-random",
                "Oracle LB": "Oracle LB", "contig-local-busy": "Local-busy", "contig-local-random": "Local-random",
                "sw-contig-local-busy": "Local-busy", "sw-contig-local-random": "Local-random",
                "inv-static": "Static", "inv-local-busy": "Local-busy", "inv-local-random": "Local-random", "inv-offload-local-single": "Offloading",
                "inv-global-busy": "Global-busy", "inv-global-random": "Global-random", "Ideal LB": "Ideal LB"}

color_dict = {"Static": "purple",
                    "Local-busy": "mediumturquoise",
                    "Local-random": "orange",
                    "Global-busy": "yellowgreen",
                    "Global-random": "orangered",
                    "Offloading": "palevioletred",
                    "Ideal LB": "lightsteelblue"}
