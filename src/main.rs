mod gpu;
mod physics;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<gpu::Gpu>,
    physics: physics::Physics,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let win = Arc::new(el.create_window(Window::default_attributes()).unwrap());
        self.gpu = Some(pollster::block_on(gpu::Gpu::new(win.clone())));
        self.window = Some(win);
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::RedrawRequested => {
                self.physics.step();
                self.gpu.as_mut().unwrap().render(&self.physics);
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop
        .run_app(&mut App { window: None, gpu: None, physics: physics::Physics::new() })
        .unwrap();
}
