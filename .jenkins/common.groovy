// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel)
        }
    }
    def command = """
                    #!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    ${getDependenciesCommand}
                    ./install -v 1
                  """
    platform.runCommand(this, command)
}

return this
