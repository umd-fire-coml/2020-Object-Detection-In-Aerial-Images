# Environment Checker Script:
# run 'conda env export | python3 environment_checker.py
# Checker will tell you if you are missing a package or have the wrong version
# %%
import sys
import re
# %%
#getting dependencies from environment.yml
env_yml = open('environment.yml', 'r')
dependencies = {}
for line in env_yml:
    #capturing packages and their versions
    match = re.match("^\s+-\s+([a-zA-Z]+)(?:=([0-9]+(?:\.[0-9]+)*)(?:\.\*)?)?\s*$", line)
    
    if match != None:
        package = match.groups(0)
        #Storing all packages in dictionary
        dependencies[package] = False

# %%
#iterating through stdin
input = sys.stdin
for line in input:
    for package in dependencies:
        #checking if line contains package name and if package was marked as existing
        if package[0] in line and dependencies[package] != True:
            #checks if version specification is necessary
            if package[1] != 0:
                if package[0]+'='+package[1] in line:
                    dependencies[package] = True
                else:
                    dependencies[package] = "Wrong Version"
            else:
                dependencies[package] = True

for package in dependencies:
    if package[1] != 0:
        print("Your Environment has ", package[0], "=", package[1], ": ", dependencies[package]) 
    else:
        print("Your Environment has ", package[0], ": ", dependencies[package])

# %%
