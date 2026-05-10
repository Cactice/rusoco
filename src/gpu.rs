use std::sync::Arc;
use nalgebra::{Matrix4, Perspective3, Point3, Vector3};
use wgpu::util::DeviceExt;
use winit::window::Window;
use crate::physics::Physics;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    nor: [f32; 3],
}

#[rustfmt::skip]
const VERTICES: &[Vertex] = &[
    Vertex { pos: [ 0.5, -0.5, -0.5], nor: [ 1.,  0.,  0.] },
    Vertex { pos: [ 0.5,  0.5, -0.5], nor: [ 1.,  0.,  0.] },
    Vertex { pos: [ 0.5,  0.5,  0.5], nor: [ 1.,  0.,  0.] },
    Vertex { pos: [ 0.5, -0.5,  0.5], nor: [ 1.,  0.,  0.] },
    Vertex { pos: [-0.5, -0.5,  0.5], nor: [-1.,  0.,  0.] },
    Vertex { pos: [-0.5,  0.5,  0.5], nor: [-1.,  0.,  0.] },
    Vertex { pos: [-0.5,  0.5, -0.5], nor: [-1.,  0.,  0.] },
    Vertex { pos: [-0.5, -0.5, -0.5], nor: [-1.,  0.,  0.] },
    Vertex { pos: [-0.5,  0.5, -0.5], nor: [ 0.,  1.,  0.] },
    Vertex { pos: [-0.5,  0.5,  0.5], nor: [ 0.,  1.,  0.] },
    Vertex { pos: [ 0.5,  0.5,  0.5], nor: [ 0.,  1.,  0.] },
    Vertex { pos: [ 0.5,  0.5, -0.5], nor: [ 0.,  1.,  0.] },
    Vertex { pos: [-0.5, -0.5,  0.5], nor: [ 0., -1.,  0.] },
    Vertex { pos: [-0.5, -0.5, -0.5], nor: [ 0., -1.,  0.] },
    Vertex { pos: [ 0.5, -0.5, -0.5], nor: [ 0., -1.,  0.] },
    Vertex { pos: [ 0.5, -0.5,  0.5], nor: [ 0., -1.,  0.] },
    Vertex { pos: [-0.5, -0.5,  0.5], nor: [ 0.,  0.,  1.] },
    Vertex { pos: [ 0.5, -0.5,  0.5], nor: [ 0.,  0.,  1.] },
    Vertex { pos: [ 0.5,  0.5,  0.5], nor: [ 0.,  0.,  1.] },
    Vertex { pos: [-0.5,  0.5,  0.5], nor: [ 0.,  0.,  1.] },
    Vertex { pos: [ 0.5, -0.5, -0.5], nor: [ 0.,  0., -1.] },
    Vertex { pos: [-0.5, -0.5, -0.5], nor: [ 0.,  0., -1.] },
    Vertex { pos: [-0.5,  0.5, -0.5], nor: [ 0.,  0., -1.] },
    Vertex { pos: [ 0.5,  0.5, -0.5], nor: [ 0.,  0., -1.] },
];

#[rustfmt::skip]
const INDICES: &[u16] = &[
     0,  1,  2,   0,  2,  3,
     4,  5,  6,   4,  6,  7,
     8,  9, 10,   8, 10, 11,
    12, 13, 14,  12, 14, 15,
    16, 17, 18,  16, 18, 19,
    20, 21, 22,  20, 22, 23,
];

const SHADER: &str = r#"
struct Uniforms { vp: mat4x4<f32> }
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertIn {
    @location(0) pos: vec3<f32>,
    @location(1) nor: vec3<f32>,
    @location(2) m0:  vec4<f32>,
    @location(3) m1:  vec4<f32>,
    @location(4) m2:  vec4<f32>,
    @location(5) m3:  vec4<f32>,
    @location(6) col: vec4<f32>,
}
struct VertOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) nor: vec3<f32>,
    @location(1) col: vec4<f32>,
}

@vertex fn vs(v: VertIn) -> VertOut {
    let m = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    return VertOut(u.vp * m * vec4<f32>(v.pos, 1.0), normalize((m * vec4<f32>(v.nor, 0.0)).xyz), v.col);
}

@fragment fn fs(f: VertOut) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(1.0, 2.0, 1.0));
    return f.col * max(dot(f.nor, light), 0.2);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

pub struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    ubuf: wgpu::Buffer,
    bg: wgpu::BindGroup,
    depth: wgpu::TextureView,
    window: Arc<Window>,
}

impl Gpu {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let inst = wgpu::Instance::default();
        let surface = inst.create_surface(window.clone()).unwrap();
        let adapter = inst
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) =
            adapter.request_device(&Default::default(), None).await.unwrap();

        let caps = surface.get_capabilities(&adapter);
        let fmt = caps.formats[0];
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        });

        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 24,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute { offset:  0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                            wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 80,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute { offset:  0, shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
                            wgpu::VertexAttribute { offset: 16, shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
                            wgpu::VertexAttribute { offset: 32, shader_location: 4, format: wgpu::VertexFormat::Float32x4 },
                            wgpu::VertexAttribute { offset: 48, shader_location: 5, format: wgpu::VertexFormat::Float32x4 },
                            wgpu::VertexAttribute { offset: 64, shader_location: 6, format: wgpu::VertexFormat::Float32x4 },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: fmt,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let depth = depth_view(&device, size.width, size.height);

        Self { surface, device, queue, pipeline, vbuf, ibuf, ubuf, bg, depth, window }
    }

    pub fn render(&mut self, physics: &Physics) {
        let sz = self.window.inner_size();
        let vp = view_proj(sz.width as f32 / sz.height as f32);
        self.queue.write_buffer(&self.ubuf, 0, bytemuck::cast_slice(vp.as_slice()));

        let floor = Matrix4::new_translation(&Vector3::new(0.0, -0.1, 0.0))
            * Matrix4::new_nonuniform_scaling(&Vector3::new(10.0, 0.2, 10.0));

        let instances = [
            Instance { model: mat4_cols(&floor),            color: [0.4, 0.4, 0.4, 1.0] },
            Instance { model: physics.box_matrix(),         color: [0.9, 0.5, 0.2, 1.0] },
        ];
        let inst_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let frame = self.surface.get_current_texture().unwrap();
        let view = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.08, g: 0.08, b: 0.1, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bg, &[]);
            pass.set_vertex_buffer(0, self.vbuf.slice(..));
            pass.set_vertex_buffer(1, inst_buf.slice(..));
            pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..INDICES.len() as u32, 0, 0..instances.len() as u32);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        frame.present();
    }
}

fn depth_view(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn view_proj(aspect: f32) -> Matrix4<f32> {
    let view = Matrix4::look_at_rh(
        &Point3::new(4.0, 3.0, 7.0),
        &Point3::new(0.0, 1.0, 0.0),
        &Vector3::y(),
    );
    #[rustfmt::skip]
    let gl_to_wgpu = Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
    );
    gl_to_wgpu * Perspective3::new(aspect, 1.0, 0.1, 100.0).to_homogeneous() * view
}

fn mat4_cols(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let s = m.as_slice();
    [
        [s[0],  s[1],  s[2],  s[3]],
        [s[4],  s[5],  s[6],  s[7]],
        [s[8],  s[9],  s[10], s[11]],
        [s[12], s[13], s[14], s[15]],
    ]
}
