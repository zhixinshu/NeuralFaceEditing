require 'nn'

local SHShading, parent = torch.class('nn.SHShading', 'nn.Module')

-- The synthesis layer using shperical harmonics

function SHShading:__init(dimension)
    -- ??? how to initialize ???
    parent.__init(self)
    self.gradInput = {}
    self.nSample = 1
    self.nPixel = 1
end

local function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function SHShading:updateOutput(input)

    -- Inputs are A (albedo), L (lighting) and N (normals)
    -- A: 3 channels image, R, G, B, (tensor size 3xWxH)
    -- N: 3 channels image, Nx, Ny, Nz, (tensor size 3xWxH)
    -- L: 9 dimensional vector, spherical harmonics coefficient (size 9)
    local N = input[1] -- num x 3 x W x H tensor
    local L = input[2] -- num x 9 tensor
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
    local M_diag = torch.Tensor(4*self.nSample,4*self.nSample):fill(0)
    for i = 1,self.nSample do
        M_diag[{{4*i-3, 4*i},{4*i-3, 4*i}}] = getMMatrix(L[i]) 
    end
    local S_group = torch.cmul(Nt*M_diag , Nt)
    
    --local S = torch.Tensor(self.nSample,3,self.nPixel)
    local S = torch.Tensor(self.nSample,3,N:size(3),N:size(4))
    for j=1,self.nSample do 
        S[{{j},{1},{}}] = torch.sum(S_group[{{},{j*4-3, j*4}}],2)
        S[{{j},{2},{}}] = S[{{j},{1},{}}] -- use same shading
        S[{{j},{3},{}}] = S[{{j},{1},{}}] -- use same shading
    end



    -- element wise multiplication of shading and albedo, I = S * A
    -- same shading applies to all albedo channel: I[1] = S*A[1], I[2] = S*A[2], I[3] = S*A[3]
    self.output = S
    return self.output
end

function SHShading:updateGradInput(input, gradOutput)
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

    local N = input[1]
    local L = input[2]
    self.nSample= N:size(1) -- number of samples in this batch
    self.nPixel = N:size(3)*N:size(4) -- number of pixel equals to W*H
    -- compute dIdA
    
    local Nx = N:narrow(2,1,1)
    local Ny = N:narrow(2,2,1)
    local Nz = N:narrow(2,3,1)

    -- compute dIdS 
    local dIdS = gradOutput

    dIdS = torch.sum(dIdS,2)/3 

    -- compute dIdN = [dIdNx, dIdNy, dIdNz], 
        -- dIdNx = dIdS * dSdNx = As * dSdNx
    local dIdN = torch.Tensor(self.nSample, 3, self.nPixel)

    for j = 1,self.nSample do
        local L_j = L[j]
        local dSdNx = 2*c1*L_j[9]*Nx[j] + 2*c1*L_j[5]*Ny[j] + 2*c1*L_j[8]*Nz[j] + 2*c2*L_j[4]
        dIdN[j][1] = torch.cmul(dIdS[j], dSdNx)    
        local dSdNy = 2*c1*L_j[5]*Nx[j] - 2*c1*L_j[9]*Ny[j] + 2*c1*L_j[6]*Nz[j] + 2*c2*L_j[2]
        dIdN[j][2] = torch.cmul(dIdS[j], dSdNy)
        local dSdNz = 2*c1*L_j[8]*Nx[j] + 2*c1*L_j[6]*Ny[j] + 2*c3*L_j[7]*Nz[j] + 2*c2*L_j[3]
        dIdN[j][3] = torch.cmul(dIdS[j], dSdNz)  
    end
    self.gradInput[1] = dIdN

    -- compute dIdL
    -- local nPixel = 1
    local dIdL = torch.Tensor(self.nSample, 9)
    for j =1,self.nSample do
        dIdL[j][1] = torch.sum(c4*dIdS[j])/self.nPixel
        dIdL[j][2] = torch.sum(torch.cmul(dIdS[j], 2*c2*Ny[j]))/self.nPixel
        dIdL[j][3] = torch.sum(torch.cmul(dIdS[j], 2*c2*Nz[j]))/self.nPixel
        dIdL[j][4] = torch.sum(torch.cmul(dIdS[j], 2*c2*Nx[j]))/self.nPixel
        dIdL[j][5] = torch.sum(torch.cmul(dIdS[j], 2*c1*torch.cmul(Nx[j],Ny[j])))/self.nPixel
        dIdL[j][6] = torch.sum(torch.cmul(dIdS[j], 2*c1*torch.cmul(Ny[j],Nz[j])))/self.nPixel
        dIdL[j][7] = torch.sum(torch.cmul(dIdS[j], c3*torch.cmul(Nz[j],Nz[j]) - c5))/self.nPixel
        dIdL[j][8] = torch.sum(torch.cmul(dIdS[j], 2*c1*torch.cmul(Nx[j],Nz[j])))/self.nPixel
        dIdL[j][9] = torch.sum(torch.cmul(dIdS[j], c1*torch.cmul(Nx[j],Nx[j]) - c1*torch.cmul(Ny[j],Ny[j])))/self.nPixel
    end
    self.gradInput[2] = dIdL

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
