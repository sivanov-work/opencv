#ifndef GAPI_STREAMING_ONEVPL_ACCEL_DX11_ALLOC_RESOURCE_HPP
#define GAPI_STREAMING_ONEVPL_ACCEL_DX11_ALLOC_RESOURCE_HPP

#include <map>

#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#include "streaming/onevpl/accelerators/utils/elastic_barrier.hpp"

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
#pragma comment(lib,"d3d11.lib")

#define D3D11_NO_HELPERS
#define NOMINMAX
#include <d3d11.h>
#include <d3d11_4.h>
#include <codecvt>
#include "opencv2/core/directx.hpp"
#ifdef HAVE_OPENCL
#include <CL/cl_d3d11.h>
#endif // HAVE_OPENCL
#undef D3D11_NO_HELPERS
#undef NOMINMAX

namespace cv {
namespace gapi {
namespace wip {

class SharedLock;
struct LockAdapter {
    size_t read_lock();
    size_t unlock_read();

    void write_lock();
    bool is_write_acquired();
    void unlock_write();

    SharedLock* set_adaptee(SharedLock* new_impl);
    SharedLock* get_adaptee();
private:
    SharedLock* impl = nullptr;
};

struct DX11AllocationRecord;
struct DX11AllocationItem : public LockAdapter,
                           public elastic_barrier<DX11AllocationItem> {
    using subresource_id_t = unsigned int;

    DX11AllocationItem(std::weak_ptr<DX11AllocationRecord> parent,
                      ID3D11Texture2D* texture_ptr,
                      subresource_id_t subresource_id,
                      ID3D11Texture2D* staging_tex_ptr);
    ~DX11AllocationItem();

    void release();
    ID3D11Texture2D* get_texture();
    ID3D11Texture2D* get_staging_texture();
    DX11AllocationItem::subresource_id_t get_subresource() const;

    // elastic barrier interface impl
    void on_first_in_impl(ID3D11DeviceContext* device_context, mfxFrameData *ptr);
    void on_last_out_impl(ID3D11DeviceContext* device_context, mfxFrameData *ptr);
private:
    ID3D11Texture2D* texture_ptr = nullptr;
    subresource_id_t subresource_id = 0;
    ID3D11Texture2D* staging_texture_ptr = nullptr;
    std::weak_ptr<DX11AllocationRecord> observer;
};

struct DX11AllocationRecord : public std::enable_shared_from_this<DX11AllocationRecord> {

    using Ptr = std::shared_ptr<DX11AllocationRecord>;

    ~DX11AllocationRecord();

    template<typename... Args>
    static Ptr create(Args&& ...args) {
        std::shared_ptr<DX11AllocationRecord> record(new DX11AllocationRecord);
        record->init(std::forward<Args>(args)...);
        return record;
    }

    Ptr get_ptr();

    // Raw ptr is required as a part of VPL `Mid` c-interface
    // which requires contiguous memory
    using AllocationId = DX11AllocationItem*;
    AllocationId* data();
private:
    DX11AllocationRecord();
    void init(unsigned int items, ID3D11Texture2D* texture, std::vector<ID3D11Texture2D*> &&staging_textures);

    std::vector<AllocationId> resources;
    ID3D11Texture2D* texture_ptr = nullptr;
};
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ACCEL_DX11_ALLOC_RESOURCE_HPP
