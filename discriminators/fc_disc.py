class FCDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[], patch=False):
        super(FCDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_sigmoid = use_sigmoid
        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                        kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if patch:
            self.linear = nn.Linear(7*7,1)
        else:
            self.linear = nn.Linear(13*13,1)
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        batchsize = input.size()[0]
        output = self.model(input)
        output = output.view(batchsize,-1)
        # print(output.size())
        output = self.linear(output)
        if self.use_sigmoid:
            print("sigmoid")
            output = self.sigmoid(output)
        return output

