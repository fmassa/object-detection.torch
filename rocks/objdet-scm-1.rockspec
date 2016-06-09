package = "objdet"
version = "scm-1"

source = {
   url = "git://github.com/fmassa/object-detection.torch",
   tag = "master"
}

description = {
   summary = "Generic framework for object detection in Torch7.",
   homepage = "git://github.com/fmassa/object-detection.torch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "graph",
   "nn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
