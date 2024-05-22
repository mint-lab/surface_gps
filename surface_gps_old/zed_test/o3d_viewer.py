import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class O3DViewer:
    def __init__(self, title='Open3D Viewer', width=1024, height=768):
        self.window = gui.Application.instance.create_window(title, width, height)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        gui.Application.instance.menubar = None

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.show_axes(True)
        self.window.add_child(self.scene)

        em = self.window.theme.font_size
        margin = 0.3 * em
        self.panel = gui.Vert(margin, gui.Margins(margin))
        self.window.add_child(self.panel)

        collapse1 = gui.CollapsableVert("Color Image", 0, gui.Margins(margin, 0, 2*margin, 0))
        self.widget_color = gui.ImageWidget()
        collapse1.add_child(self.widget_color)
        self.panel.add_child(collapse1)
        collapse2 = gui.CollapsableVert("Depth Image", 0, gui.Margins(margin, 0, 2*margin, 0))
        self.widget_depth = gui.ImageWidget()
        collapse2.add_child(self.widget_depth)
        self.panel.add_child(collapse2)

        self.is_closed = False

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        panel_width = 15 * layout_context.theme.font_size
        self.scene.frame = gui.Rect(contentRect.x, contentRect.y, contentRect.width - panel_width, contentRect.height)
        self.panel.frame = gui.Rect(self.scene.frame.get_right(), contentRect.y, panel_width, contentRect.height)

    def _on_close(self):
        self.is_closed = True
        return True



if __name__ == '__main__':
    app = gui.Application.instance
    app.initialize()
    win = O3DViewer()
    app.run()
