
	using Images
	using Flux, Flux.Data
	using Flux: onehotbatch, throttle, crossentropy, @epochs
	using MLDataUtils
	using Base.Iterators: partition, repeated
	using Printf
	using Statistics
	using Plots

function resize_and_grayify(directory, im_name, width::Int64, height::Int64)
    resized_gray_img = Gray.(load(directory * "/" * im_name)) |> (x -> imresize(x, width, height))
    try
        save("preprocessed_" * directory * "/" * im_name, resized_gray_img)
    catch e
        if isa(e, SystemError)
            mkdir("preprocessed_" * directory)
            save("preprocessed_" * directory * "/" * im_name, resized_gray_img)
        end
    end
end

function process_images(directory, width::Int64, height::Int64)
    files_list = readdir(directory)
    map(x -> resize_and_grayify(directory, x, width, height),                               files_list)
end

n_resolution = 90

begin
	process_images("Data/Train/Positive", n_resolution, n_resolution)
	process_images("Data/Train/Negative", n_resolution, n_resolution)
end

begin
	positive_dir = readdir("preprocessed_Data/Train/Positive")
	negative_dir = readdir("preprocessed_Data/Train/Negative")
end

begin
	# we load the pre-proccessed images
	pos = load.("preprocessed_Data/Train/Positive/" .* positive_dir)
	neg = load.("preprocessed_Data/Train/Negative/" .* negative_dir)
end

composed_data = vcat(pos, neg)

begin
    labels = vcat([0 for _ in 1:length(neg)], [1 for _ in 1:length(pos)])
    (x_train, y_train), (x_test, y_test) = splitobs(shuffleobs((composed_data, labels)), at = 0.7)
end

x_train
x_test


# this function creates minibatches of the data. Each minibatch is a tuple of (data, labels).
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:1)
    return (X_batch, Y_batch)
end

begin
    # here we define the train and test sets.
    batchsize = 128
    mb_idxs = partition(1:length(x_train), batchsize)
    train_set = [make_minibatch(x_train, y_train, i) for i in mb_idxs]
    test_set = make_minibatch(x_test, y_test, 1:length(x_test));
end

Flux.flatten(train_set)



model = Chain(
		Flux.flatten,
		Dense(8100, 90, relu),
        Dense(90, 32, relu),
        Dense(32, 2),
        softmax)




train_loss = Float64[]
test_loss = Float64[]
acc = Float64[]
ps = Flux.params(model)
opt = ADAM()
L(x, y) = Flux.crossentropy(model(x), y)
L(x, y) = Flux.crossentropy(model(x), y)
accuracy(x, y, f) = mean(Flux.onecold(f(x)) .== Flux.onecold(y))

function update_loss!()
    push!(train_loss, mean(L(train_set)))
    push!(test_loss, mean(L.(test_set)))
    push!(acc, accuracy(test_set..., model))
    @printf("train loss = %.2f, test loss = %.2f, accuracy = %.2f\n", train_loss[end], test_loss[end], acc[end])
end


@epochs 100 Flux.train!(L, ps, train_set, opt;
               cb = Flux.throttle(update_loss!, 5))
begin
   plot(train_loss, xlabel="Epochs", title="Model Training", label="Train loss", lw=2, alpha=0.9)
   plot!(test_loss, label="Test loss", lw=2, alpha=0.9)
   plot!(acc, label="Accuracy", lw=2, alpha=0.9)
end
