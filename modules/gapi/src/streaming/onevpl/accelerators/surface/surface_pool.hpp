#ifndef GAPI_STREAMING_ONEVPL_SURFACE_SURFACE_POOL_HPP
#define GAPI_STREAMING_ONEVPL_SURFACE_SURFACE_POOL_HPP

#include <map>
#include <memory>
#include <vector>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

#ifdef HAVE_ONEVPL
#if (MFX_VERSION >= 2000)
#include <vpl/mfxdispatcher.h>
#endif

#include <vpl/mfx.h>

namespace cv {
namespace gapi {
namespace wip {

class Surface;
class CachedPool {
public:
    using surface_ptr_t = std::shared_ptr<Surface>;
    using surface_container_t = std::vector<surface_ptr_t>;
    using free_surface_iterator_t = typename surface_container_t::iterator;
    using cached_surface_container_t = std::map<mfxFrameSurface1*, surface_ptr_t>;

    // GAPI_EXPORTS for tests
    GAPI_EXPORTS void push_back(surface_ptr_t &&surf);
    GAPI_EXPORTS void reserve(size_t size);
    GAPI_EXPORTS size_t total_size() const;
    GAPI_EXPORTS size_t available_size() const;
    GAPI_EXPORTS void clear();

    GAPI_EXPORTS surface_ptr_t find_free();
    GAPI_EXPORTS surface_ptr_t find_by_handle(mfxFrameSurface1* handle);
private:
    surface_container_t surfaces;
    free_surface_iterator_t next_free_it;
    cached_surface_container_t cache;
};
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_SURFACE_SURFACE_POOL_HPP
