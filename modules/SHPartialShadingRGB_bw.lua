require 'nn'

local SHPartialShadingRGB_bw, parent = torch.class('nn.SHPartialShadingRGB_bw', 'nn.Module')

-- The synthesis layer using shperical harmonics

function SHPartialShadingRGB_bw:__init(dimension)
    -- ??? how to initialize ???
    parent.__init(self)
    self.gradInput = {}
    self.nSample = 1
    self.nPixel = 1
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function SHPartialShadingRGB_bw:updateOutput(input)

    -- Inputs are A (albedo), L (lighting) and N (normals)
    -- A: 3 channels image, R, G, B, (tensor size 3xWxH)
    -- N: 3 channels image, Nx, Ny, Nz, (tensor size 3xWxH)
    -- L: 9 dimensional vector, spherical harmonics coefficient (size 9)
    local N = input[1] -- num x 3 x W x H tensor
    local Lr_bw = input[2] -- num x 9 tensor
    local Lr = Lr_bw[{{},{1,9}}]
    local Lg_bw = input[3] -- num x 9 tensor
    local Lg = Lg_bw[{{},{1,9}}]
    local Lb_bw = input[4] -- num x 9 tensor
    local Lb = Lb_bw[{{},{1,9}}]
    local mask = input[5] --  num x 3 x W x H mask
    --print(N:size())
    self.nSample= N:size(1) -- number of samples in this batch
    self.nPixel = N:size(3)*N:size(4) -- number of pixel equals to W*H
    -- Output is synthesized image, R, G, B (tensor size 3xWxH)
    -- generate shading from normals and albedo
    local Ns = torch.reshape(N, self.nSample, 3, self.nPixel)
    local N_ext = torch.Tensor(self.nSample, 1, self.nPixel):fill(1)
    Ns = torch.cat(Ns, N_ext, 2) -- [Nx, Ny, Nz, 1]
    -- convert to Matrix, [Nx_1,Ny_1,Nx_1, 1, Nx_2, Ny_2,Nz_2, 1,...., Nz_nSample,1]
    Nt = Ns:reshape(self.nSample*4,self.nPixel):t()  
    
    -- generate M_diag, M matrces on its diag
    local M_diag_r = torch.Tensor(4*self.nSample,4*self.nSample):fill(0)
    local M_diag_g = torch.Tensor(4*self.nSample,4*self.nSample):fill(0)
    local M_diag_b = torch.Tensor(4*self.nSample,4*self.nSample):fill(0)
    for i = 1,self.nSample do
        M_diag_r[{{4*i-3, 4*i},{4*i-3, 4*i}}] = getMMatrix(Lr[i]) 
        M_diag_g[{{4*i-3, 4*i},{4*i-3, 4*i}}] = getMMatrix(Lg[i]) 
        M_diag_b[{{4*i-3, 4*i},{4*i-3, 4*i}}] = getMMatrix(Lb[i]) 
    end
    --local S_group = torch.cmul(Nt*M_diag , Nt)
    local Sr = torch.cmul(Nt*M_diag_r , Nt)
    local Sg = torch.cmul(Nt*M_diag_g , Nt)
    local Sb = torch.cmul(Nt*M_diag_b , Nt)
    
    --local S = torch.Tensor(self.nSample,3,self.nPixel)
    local S = torch.Tensor(self.nSample,3,N:size(3),N:size(4))
    for j=1,self.nSample do 
        S[{{j},{1},{}}] = torch.sum(Sr[{{},{j*4-3, j*4}}],2)*Lr_bw[j][10]
        S[{{j},{2},{}}] = torch.sum(Sg[{{},{j*4-3, j*4}}],2)*Lg_bw[j][10]
        S[{{j},{3},{}}] = torch.sum(Sb[{{},{j*4-3, j*4}}],2)*Lb_bw[j][10]
    end



    -- element wise multiplication of shading and albedo, I = S * A
    -- same shading applies to all albedo channel: I[1] = S*A[1], I[2] = S*A[2], I[3] = S*A[3]
    self.output = S
    return self.output
end

function SHPartialShadingRGB_bw:updateGradInput(input, gradOutput)
    -- Verify again for correct handling of 0.5 multiplication
    self.gradInput = {}
    -- This is a layer with no parameter, gradInput = gradOutput*f'(x)
    -- self.gradInput[1] : dIdA = S, should be size of A, therefore basically [S S S]
    -- self.gradInput[2] : dIdN = dIdS * dSdN = A * dSdN, should be size of N, element wise operation
    -- self.gradInput[3] : dIdL = dIdS * dSdL = A * dSdL, should be size of L, element wise(and normalize)
    local c1 = 0.429043
    local c2 = 0.511664
    local c3 = 0.743152
    local c4 = 0.886227
    local c5 = 0.247708

    local N = input[1] -- num x 3 x W x H tensor
    local Lr_bw = input[2] -- num x 9 tensor
    local Lr = Lr_bw[{{},{1,9}}]
    local Lg_bw = input[3] -- num x 9 tensor
    local Lg = Lg_bw[{{},{1,9}}]
    local Lb_bw = input[4] -- num x 9 tensor
    local Lb = Lb_bw[{{},{1,9}}]
    local mask = input[5] --  num x 3 x W x H mask
    self.nSample= N:size(1) -- number of samples in this batch
    self.nPixel = N:size(3)*N:size(4) -- number of pixel equals to W*H
    -- compute dIdA
    
    local Nx = N:narrow(2,1,1)
    local Ny = N:narrow(2,2,1)
    local Nz = N:narrow(2,3,1)

    -- compute dIdS 
    local dIdS = gradOutput

    --dIdS = torch.sum(dIdS,2)/3 
    local dIdSr = dIdS[{{},{1},{}}]
    local dIdSg = dIdS[{{},{2},{}}]
    local dIdSb = dIdS[{{},{3},{}}]

    local mask_single = mask:narrow(2,1,1)


    -- compute dIdN = [dIdNx, dIdNy, dIdNz], 
        -- dIdNx = dIdS * dSdNx = As * dSdNx
    local dIdN = torch.Tensor(self.nSample, 3, self.nPixel)

    for j = 1,self.nSample do
        local Lr_j = Lr[j]
        local Lg_j = Lg[j]
        local Lb_j = Lb[j]
        local dSrdNx = Nx[j]*Lr_j[9]*c1*2 + Ny[j]*Lr_j[5]*c1*2 + Nz[j]*Lr_j[8]*c1*2 + Lr_j[4]*c2*2
        local dSgdNx = Nx[j]*Lg_j[9]*c1*2 + Ny[j]*Lg_j[5]*c1*2 + Nz[j]*Lg_j[8]*c1*2 + Lg_j[4]*c2*2
        local dSbdNx = Nx[j]*Lb_j[9]*c1*2 + Ny[j]*Lb_j[5]*c1*2 + Nz[j]*Lb_j[8]*c1*2 + Lb_j[4]*c2*2
        dIdN[j][1] = torch.cmul(dIdSr[j], dSrdNx) + torch.cmul(dIdSg[j], dSgdNx) + torch.cmul(dIdSb[j], dSbdNx)     
        local dSrdNy = Nx[j]*Lr_j[5]*c1*2 - Ny[j]*Lr_j[9]*c1*2 + Nz[j]*Lr_j[6]*c1*2 + Lr_j[2]*2*c2
        local dSgdNy = Nx[j]*Lg_j[5]*c1*2 - Ny[j]*Lg_j[9]*c1*2 + Nz[j]*Lg_j[6]*c1*2 + Lg_j[2]*2*c2
        local dSbdNy = Nx[j]*Lb_j[5]*c1*2 - Ny[j]*Lb_j[9]*c1*2 + Nz[j]*Lb_j[6]*c1*2 + Lb_j[2]*2*c2
        dIdN[j][2] = torch.cmul(dIdSr[j], dSrdNy) + torch.cmul(dIdSg[j], dSgdNy) + torch.cmul(dIdSb[j], dSbdNy)
        local dSrdNz = Nx[j]*Lr_j[8]*c1*2 + Ny[j]*Lr_j[6]*c1*2 + Nz[j]*Lr_j[7]*c3*2 + Lr_j[3]*2*c2
        local dSgdNz = Nx[j]*Lg_j[8]*c1*2 + Ny[j]*Lg_j[6]*c1*2 + Nz[j]*Lg_j[7]*c3*2 + Lg_j[3]*2*c2
        local dSbdNz = Nx[j]*Lb_j[8]*c1*2 + Ny[j]*Lb_j[6]*c1*2 + Nz[j]*Lb_j[7]*c3*2 + Lb_j[3]*2*c2
        dIdN[j][3] = torch.cmul(dIdSr[j], dSrdNz) + torch.cmul(dIdSg[j], dSgdNz) + torch.cmul(dIdSb[j], dSbdNz)  
    end

    self.gradInput[1] = dIdN

    -- compute dIdL
    -- local nPixel = 1
    local eps = 1e-8
    local dIdLr = torch.Tensor(self.nSample, 10)
    local dIdLg = torch.Tensor(self.nSample, 10)
    local dIdLb = torch.Tensor(self.nSample, 10)
    for j = 1,self.nSample do
        --local dIdS_partial = torch.cmul(dIdS[j],mask_single[j])
        local dIdSr_partial = torch.cmul(dIdSr[j],mask_single[j])
        local dIdSg_partial = torch.cmul(dIdSg[j],mask_single[j])
        local dIdSb_partial = torch.cmul(dIdSb[j],mask_single[j])
        local nPixel_partial = mask_single[j]:sum() + eps
        local Nx_partial = torch.cmul(Nx[j],mask_single[j])
        local Ny_partial = torch.cmul(Ny[j],mask_single[j])
        local Nz_partial = torch.cmul(Nz[j],mask_single[j])
        -- Lr gradient
        dIdLr[j][1] = torch.sum(dIdSr_partial*c4)/nPixel_partial
        dIdLr[j][2] = torch.sum(torch.cmul(dIdSr_partial , Ny_partial*2*c2))/nPixel_partial
        dIdLr[j][3] = torch.sum(torch.cmul(dIdSr_partial , Nz_partial*2*c2))/nPixel_partial
        dIdLr[j][4] = torch.sum(torch.cmul(dIdSr_partial , Nx_partial*2*c2))/nPixel_partial
        dIdLr[j][5] = torch.sum(torch.cmul(dIdSr_partial , torch.cmul(Nx_partial,Ny_partial)*2*c1))/nPixel_partial
        dIdLr[j][6] = torch.sum(torch.cmul(dIdSr_partial , torch.cmul(Ny_partial,Nz_partial)*2*c1))/nPixel_partial
        dIdLr[j][7] = torch.sum(torch.cmul(dIdSr_partial , torch.cmul(Nz_partial,Nz_partial)*c3 - c5))/nPixel_partial
        dIdLr[j][8] = torch.sum(torch.cmul(dIdSr_partial , torch.cmul(Nx_partial,Nz_partial)*2*c1))/nPixel_partial
        dIdLr[j][9] = torch.sum(torch.cmul(dIdSr_partial , torch.cmul(Nx_partial,Nx_partial)*c1 - torch.cmul(Ny_partial,Ny_partial)*c1))/nPixel_partial
        -- Lg gradient
        dIdLg[j][1] = torch.sum(dIdSg_partial*c4)/nPixel_partial
        dIdLg[j][2] = torch.sum(torch.cmul(dIdSg_partial , Ny_partial*2*c2))/nPixel_partial
        dIdLg[j][3] = torch.sum(torch.cmul(dIdSg_partial , Nz_partial*2*c2))/nPixel_partial
        dIdLg[j][4] = torch.sum(torch.cmul(dIdSg_partial , Nx_partial*2*c2))/nPixel_partial
        dIdLg[j][5] = torch.sum(torch.cmul(dIdSg_partial , torch.cmul(Nx_partial,Ny_partial)*2*c1))/nPixel_partial
        dIdLg[j][6] = torch.sum(torch.cmul(dIdSg_partial , torch.cmul(Ny_partial,Nz_partial)*2*c1))/nPixel_partial
        dIdLg[j][7] = torch.sum(torch.cmul(dIdSg_partial , torch.cmul(Nz_partial,Nz_partial)*c3 - c5))/nPixel_partial
        dIdLg[j][8] = torch.sum(torch.cmul(dIdSg_partial , torch.cmul(Nx_partial,Nz_partial)*2*c1))/nPixel_partial
        dIdLg[j][9] = torch.sum(torch.cmul(dIdSg_partial , torch.cmul(Nx_partial,Nx_partial)*c1 - torch.cmul(Ny_partial,Ny_partial)*c1))/nPixel_partial
        -- Lb gradient
        dIdLb[j][1] = torch.sum(dIdSb_partial*c4)/nPixel_partial
        dIdLb[j][2] = torch.sum(torch.cmul(dIdSb_partial , Ny_partial*2*c2))/nPixel_partial
        dIdLb[j][3] = torch.sum(torch.cmul(dIdSb_partial , Nz_partial*2*c2))/nPixel_partial
        dIdLb[j][4] = torch.sum(torch.cmul(dIdSb_partial , Nx_partial*2*c2))/nPixel_partial
        dIdLb[j][5] = torch.sum(torch.cmul(dIdSb_partial , torch.cmul(Nx_partial,Ny_partial)*2*c1))/nPixel_partial
        dIdLb[j][6] = torch.sum(torch.cmul(dIdSb_partial , torch.cmul(Ny_partial,Nz_partial)*2*c1))/nPixel_partial
        dIdLb[j][7] = torch.sum(torch.cmul(dIdSb_partial , torch.cmul(Nz_partial,Nz_partial)*c3 - c5))/nPixel_partial
        dIdLb[j][8] = torch.sum(torch.cmul(dIdSb_partial , torch.cmul(Nx_partial,Nz_partial)*2*c1))/nPixel_partial
        dIdLb[j][9] = torch.sum(torch.cmul(dIdSb_partial , torch.cmul(Nx_partial,Nx_partial)*c1 - torch.cmul(Ny_partial,Ny_partial)*c1))/nPixel_partial
        -- multiplier gradient
        dIdLr[j][10] = torch.sum(dIdSr_partial)/nPixel_partial
        dIdLg[j][10] = torch.sum(dIdSg_partial)/nPixel_partial
        dIdLb[j][10] = torch.sum(dIdSb_partial)/nPixel_partial
    end 
    self.gradInput[2] = dIdLr
    self.gradInput[3] = dIdLg
    self.gradInput[4] = dIdLb
    self.gradInput[5] = dIdS*0 -- no error needed for the mask

    return self.gradInput
end

function getMMatrix(L)

    -- M = [ c1*L9   c1*L5    c1*L8   c2*L4
    --       c1*L5   -c1*L9   c1*L6   c2*L2
    --       c1*L8   c1*L6    c3*L7   c2*L3
    --       c2*L4   c2*L2    c2*L3   c4*L1 - c5*L7 ]
    local c1 = 0.429043
    local c2 = 0.511664
    local c3 = 0.743152
    local c4 = 0.886227
    local c5 = 0.247708
    local M = torch.Tensor(4,4)

    M[1][1] = c1*L[9]
    M[1][2] = c1*L[5]
    M[1][3] = c1*L[8]
    M[1][4] = c2*L[4]

    M[2][1] = c1*L[5] 
    M[2][2] = - c1*L[9]
    M[2][3] = c1*L[6]
    M[2][4] = c2*L[2]

    M[3][1] = c1*L[8]
    M[3][2] = c1*L[6]
    M[3][3] = c3*L[7]
    M[3][4] = c2*L[3]

    M[4][1] = c2*L[4]
    M[4][2] = c2*L[2]
    M[4][3] = c2*L[3]
    M[4][4] = c4*L[1] - c5*L[7]

    return M

end
