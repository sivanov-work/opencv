// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "streaming/onevpl/onevpl_file_data_provider.hpp"

namespace cv {
namespace gapi {
namespace wip {

FileDataProvider::FileDataProvider(const std::string& file_path) :
    source_handle(fopen(file_path.c_str(), "rb"), &fclose) {
    if (!source_handle) {
        throw DataProviderSystemErrorException(errno,
                                               "FileDataProvider: cannot open source file: " + file_path);
    }
}

FileDataProvider::~FileDataProvider() = default;

size_t FileDataProvider::provide_data(size_t out_data_bytes_size, void* out_data) {
    if (empty()) {
        return 0;
    }

    size_t ret = fread(out_data, 1, out_data_bytes_size, source_handle.get());
    if (ret == 0) {
        if (feof(source_handle.get())) {
            source_handle.reset();
        } else {
            throw DataProviderSystemErrorException (errno, "FileDataProvider::provide_data error read");
        }
    }
    return ret;
}

bool FileDataProvider::empty() const {
    return !source_handle;
}
} // namespace wip
} // namespace gapi
} // namespace cv
