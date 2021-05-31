include("grasta.jl")
include("larb.jl")
using LinearAlgebra

#lr for updates 
η = .000001
ρ = 1

#Image dimensions, height and width
m=320
p=480

#image noise (stdev)
γ = .01

#Vector dimension (pixels) in image.
n=m*p

#Subspace rank.
d=9

#subsample percent
subsampling = .1

#percent of space taken by foreground
fg_percent = .05
#foreground amplitude
fg_amp = .5

#total number of frames fed into the model
numIters = 60

#number of inner iterations per U updates
iters = 10

data = rand(n,d)
U = rand(n,d)

#qr factorization of each for initial guess / true subspaces

data,_ = qr(data)
U,_ = qr(U)

#expliticly cast to array of float
data = Array{Float64}(data)
U = Array{Float64}(U)

w = zeros(d)

for i in 1:numIters

    rand_weight = rand(d)
    v = data * rand_weight
    v .+= (rand(n) .< fg_percent) * fg_amp
    v .+= randn(n) * γ


    mask = rand(n) .< subsampling

    grasta_step(U,v,w,η,ρ,iters,mask;assume_orthogonal=false)

    #println(norm(U * w .- v,1))
end