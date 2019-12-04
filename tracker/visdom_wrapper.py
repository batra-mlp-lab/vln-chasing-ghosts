import numpy as np
import visdom


class VisdomVisualize():
    def __init__(self,
                 env_name='main',
                 port=8097,
                 server="http://localhost",
                 win_prefix=""):
        '''
            Initialize a visdom server on $server:$port

            Override port and server using the local configuration from
            the json file at $config_file (containing a dict with optional
            keys 'server' and 'port').
        '''
        print("Initializing visdom env [%s]" % env_name)
        self.viz = visdom.Visdom(
                            port=port,
                            env=env_name,
                            server=server
            )
        self.env_name = env_name
        self.wins = {}
        self.win_prefix = win_prefix
        self.toTensor = lambda x: x.cpu().detach().numpy()

    @staticmethod
    def clipValue(val, max=1e12, min=-1e12):
        '''
            Preventing extremely large values from crashing plots.
            Set min/max according to expected range of sane values.
        '''
        if val != val:  # NaN value check
            return 0
        else:
            return np.clip(val, min, max)

    def line(self, x, y, key, line_name, x_label="Iterations", y_label="Loss", title=""):
        '''
            Add or update a plot on the visdom server self.viz
            Argumens:
                x : Scalar -> X-coordinate on plot
                y : Scalar -> Value at x
                key : Name of plot/graph
                line_name : Name of line within plot/graph
                xlabel : Label for x-axis (default: # Iterations)
                title: Title of the plot

            Plots and lines are created if they don't exist, otherwise
            they are updated.
        '''
        key = self.win_prefix + key

        if key in self.wins.keys():
            self.viz.line(
                X=np.array([x]),
                Y=np.array([self.clipValue(y)]),
                win=self.wins[key],
                update='append',
                name=line_name,
                opts=dict(showlegend=True),
            )
        else:
            self.wins[key] = self.viz.line(
                    X=np.array([x]),
                    Y=np.array([y]),
                    win=key,
                    update='append',
                    name=line_name,
                    opts={
                        'xlabel': x_label,
                        'ylabel': y_label,
                        'title': title,
                        'showlegend': True,
                    }
                )

    def images(self, imgs, key, caption="Images",
               maxToDisplay=9, nrow=3):
        '''
            Add images to display in a grid
            Arguments:
                imgs: List of tensors or a single image tensor of
                    shape (B x C x H x W)
                key: Name of window
                caption: Header of window
                maxToDisplay: Max images to display
        '''
        key = self.win_prefix + key
        if isinstance(imgs, list):
            if hasattr(imgs[0], "cpu"):
                imgs = map(self.toTensor, imgs)
            imagesTensor = np.stack([np.array(x)
                                    for x in imgs],0)
        else:
            if hasattr(imgs, "cpu"):
                imgs = self.toTensor(imgs)
            imagesTensor = np.array(imgs)

        if imagesTensor.shape[0] > maxToDisplay:
            imagesTensor = imagesTensor[:maxToDisplay]

        if imagesTensor.shape[1] > 3:
            raise RuntimeError("Channel dimension > 3")
        elif imagesTensor.shape[1] == 3:
            imagesTensor = utils.toRGB(imagesTensor)
            # Implement a utility here to convert to 8-bit RGB tensors
        else:
            assert False

        win = self.wins[key] if key in self.wins else None
        self.wins[key] = self.viz.images(
            imagesTensor,
            nrow=nrow,
            win=win,
            opts=dict(
                    caption=caption,
            )
        )

    def text(self, text):
        self.viz.text(text)

    def save(self):
        print("Saving visdom env to disk: {}".format(self.env_name))
        self.viz.save([self.env_name])
