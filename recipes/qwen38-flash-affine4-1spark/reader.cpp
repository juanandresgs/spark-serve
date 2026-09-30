// Independently authored bounded parallel positional reader. No model math.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <string>
#include <thread>
#include <vector>
namespace py = pybind11;
using I64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
py::array_t<uint8_t> read_rows(I64 descriptors, I64 positions, int width, int workers) {
    auto fd = descriptors.request(), off = positions.request();
    if (fd.ndim != 1 || off.ndim != 1 || fd.size != off.size || width <= 0 || workers < 1 || workers > 32)
        throw std::invalid_argument("invalid positional-read layout");
    py::array_t<uint8_t> output({fd.size, static_cast<py::ssize_t>(width)});
    auto dest = output.mutable_data();
    auto fds = static_cast<int64_t*>(fd.ptr), offsets = static_cast<int64_t*>(off.ptr);
    std::atomic<py::ssize_t> next{0};
    std::atomic<int> error{0};
    {
        py::gil_scoped_release release;
        std::vector<std::thread> threads;
        auto work = [&] {
            while (!error.load()) {
                auto row = next.fetch_add(1);
                if (row >= fd.size) return;
                if (offsets[row] < 0) { error.store(EINVAL); return; }
                size_t filled = 0;
                while (filled < static_cast<size_t>(width)) {
                    auto got = pread(static_cast<int>(fds[row]), dest + row * width + filled,
                                     width - filled, offsets[row] + filled);
                    if (got < 0 && errno == EINTR) continue;
                    if (got <= 0) { error.store(got == 0 ? EIO : errno); return; }
                    filled += got;
                }
            }
        };
        try { for (int i=0; i<workers; ++i) threads.emplace_back(work); }
        catch (...) { error.store(EAGAIN); for(auto& t:threads)t.join(); throw; }
        for(auto& t:threads)t.join();
    }
    if (error.load()) throw std::runtime_error("positional read failed, errno=" + std::to_string(error.load()));
    return output;
}
PYBIND11_MODULE(spark_positional_reader, m) { m.def("read_rows", &read_rows); }
