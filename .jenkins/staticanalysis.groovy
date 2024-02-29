#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path



def runCI =
{
    nodeDetails, jobName->

    def prj  = new rocProject('rocBLAS', 'StaticAnalysis')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false
    boolean staticAnalysis = true

    buildProject(prj, formatCheck, nodes.dockerArray, null, null, null, staticAnalysis)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def jobNameList = ["main":([ubuntu22:['any']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    def propertyList = ["main":[pipelineTriggers([cron('0 1 * * 0')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    stage(urlJobName) {
        runCI([ubuntu18:['any']], urlJobName)
    }
}
