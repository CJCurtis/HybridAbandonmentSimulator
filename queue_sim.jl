#=using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Plots")
Pkg.add("Distributions")
Pkg.add("LaTeXStrings")
Pkg.add("Random")
Pkg.add("StatsBase")
Pkg.add("CSV")
Pkg.add("HypothesisTests")
Pkg.add("Dates")=#
# using Pkg
# Pkg.add("Serialization")
using Random, Distributions, Plots, LaTeXStrings, CSV, DataFrames, StatsBase, CSV, HypothesisTests, Dates, Serialization #, Roots, ForwardDiff

function emp_distn(λ, μ1, μ2, θ, max_jobs=10^8, update_every=10^7, norm=true)#pds = 10^6, t_max = Inf, max_jobs = Inf, update_every=10^5)
    #println("Starting busy period simulation with λ: $λ, μ1: $μ1, μ2: $μ2, θ: $θ")
    pd_num = 0
    jobs_completed = 0

    states = Dict{Tuple{Int,Int}, Float64}()
    init_time = time()
    t = 0.0
    q1 = 1
    q2 = 0
    while jobs_completed < max_jobs
        #println("Current time: $t, Current queue length: $y")
        arr_rate = λ
        serv1_rate = q1 > 0 ? μ1 : 0
        serv2_rate = q2 > 0 ? μ2 : 0
        aband_rate = q1 > 0 ? (q1-1)*θ : 0
        rate = arr_rate + serv1_rate + serv2_rate + aband_rate
        dt = rand(Exponential(1/rate))
        if haskey(states, (q1, q2)) == false
            states[(q1, q2)] = 0.0
        end
        states[(q1, q2)] += dt


        # nxt_event = sample([1, 2, 3, 4], Weights([λ, (q1 > 0 ? μ1 : 0), (q2 > 0 ? μ2 : 0), (q1 > 0 ? (q1-1)*θ : 0)]))
        u = rand()
        if u < arr_rate/rate
            # Arrival
            if q1 == 0 && q2 == 0
                #=pd_num += 1
                if pd_num % update_every == 0
                    println("Starting busy period number: $pd_num at time $t")
                    println("Progress: $(round(pd_num/pds*100, digits=2))%")
                    println("Elapsed time: $(round(time() - init_time, digits=2)) seconds")
                    println("----------")
                end=#
                #println("Starting new busy period: $pd_num")
            end
            q1 += 1
        elseif u< (arr_rate + serv1_rate)/rate
            # Service completion at server 1
            q1 -= 1
            q2 += 1
        elseif u < (arr_rate + serv1_rate + serv2_rate)/rate
            # Service completion at server 2
            q2 -= 1
            jobs_completed += 1
            if jobs_completed % update_every == 0
                println("Jobs completed: $jobs_completed at time $t")
                println("Progress: $(round(jobs_completed/max_jobs*100, digits=2))%")
                println("Elapsed time: $(round(time() - init_time, digits=2)) seconds")
                println("----------")
            end
        else
            # Abandonment from queue 1
            q1 -= 1
        end
        t += dt
        
    end
    #Normalize the times to get probabilities
    sm = 1
    if norm == true
        sm = sum(v for (k,v) in states)
    end
    state_probs = Dict(k => v/sm for (k, v) in states)
    return state_probs
end

function mean_arrival_rate(λ, μ1, θ)
    pisdivp0 = ones(10000)
    pisdivp0[1] = λ/(μ1)
    for i in 2:10000
        pisdivp0[i] = pisdivp0[i-1]*(λ) / (μ1 + θ * (i-1) )
    end
    pi0 = 1/(1 + sum(pisdivp0))
    return (1-pi0) * μ1
end

function max_lambda(μ1, μ2, θ, ρ=1, tolerance=1e-3)
    lam = 0
    step = 1
    dif =  ρ*μ2 - mean_arrival_rate(lam, μ1, θ)
    while abs(dif) > tolerance
        if dif >= step
            lam += step
        else
            step /= 2
        end
        dif =  ρ*μ2 - mean_arrival_rate(lam, μ1, θ)
    end
    return lam
end

function M_fn(λ, θ, μ1, μ2)
    maxλ = λ
    pis = ones(1, 1000)
    for i in 1:999
        pis[1, i+1] = pis[1, i] * maxλ/(μ1 + θ * i)
    end
    pis = pis ./ sum(pis)
    # println("Steady-state probabilities: ", pis)

    imf = zeros(1000, 1000)
    for i in 1:1000
        imf[i, i] = μ2
        if i > 1
            imf[i, i-1] = -μ1
        end
    end

    P = zeros(1000, 1000)
    P[1, 1] = - maxλ
    P[1, 2] = maxλ
    for i in 2:999
        P[i, i+1] = maxλ
        P[i, i-1] = (i-2)*θ + μ1
        P[i, i] = -P[i, i+1] - P[i, i-1]
    end
    P[1000, 1000] = -998*θ - μ1
    P[1000, 999] = 998*θ + μ1

    A = P[2:1000, 2:1000]
    A_inv = inv(A)

    R = zeros(1000, 1000)
    R[2:1000, 2:1000] = A_inv

    eT = ones(1000, 1)

    local M = (1- μ2*(pis*imf*R*imf*R*eT)[1,1])
    return M
end

function new_M_fn(λ, θ, μ1, μ2)
    maxλ = λ
    n_states = 1000
    pis = ones(1, n_states)
    for i in 1:(n_states-1)
        pis[1, i+1] = pis[1, i] * maxλ/(μ1 + θ * (i-1))
    end
    pis = pis ./ sum(pis)
    # println("Steady-state probabilities: ", pis)

    imf = zeros(n_states, n_states)
    for i in 1:n_states
        imf[i, i] = μ2
        if i > 1
            imf[i, i-1] = -μ1
        end
    end

    P = zeros(n_states, n_states)
    P[1, 1] = -maxλ
    P[1, 2] = maxλ
    for i in 2:(n_states-1)
        P[i, i+1] = maxλ
        P[i, i-1] = (i-2)*θ + μ1
        P[i, i] = -P[i, i+1] - P[i, i-1]
    end
    P[n_states, n_states] = -(n_states-2)*θ - μ1
    P[n_states, n_states-1] = (n_states-2)*θ + μ1

    A = P[2:n_states, 2:n_states]
    A_inv = inv(A)

    R = zeros(n_states, n_states)
    R[2:n_states, 2:n_states] = A_inv

    eT = ones(n_states, 1)

    local M = (1- μ2*(pis*imf*R*imf*eT)[1,1])
    return M
end

function compute_MKS(emp_dist, λ, θ, μ1, μ2, cap=1000)
    # Marginal Q2
    marginal_q2 = Dict(k[2] => 0.0 for k in keys(emp_dist))
    for (k,v) in emp_dist
        marginal_q2[k[2]] += v
    end

    ρ = mean_arrival_rate(λ, μ1, θ) / μ2

    max_q2 = maximum(collect(keys(marginal_q2)))

    maxλ =  max_lambda(μ1, μ2, θ, 1)
    M = new_M_fn(maxλ, θ, μ1, μ2)

    exp_dist = Exponential(M)
    ks_stat = 0.0
    emp_surv_fn = 1.0
    for k in 0:max_q2-1
        if (1-ρ)*k > cap
            break
        end
        emp_surv_fn -= get(marginal_q2, k, 0.0)

        exp_surv_fn_left = 1.0 - cdf(exp_dist, (1-ρ)*k)
        exp_surv_fn_right = 1.0 - cdf(exp_dist, (1-ρ)*(k+1))

        ks_stat = max(ks_stat, abs(log(exp_surv_fn_left/ emp_surv_fn)))
        ks_stat = max(ks_stat, abs(log(exp_surv_fn_right/ emp_surv_fn)))

    end 
    return ks_stat
end

function compute_KS(emp_dist, λ, θ, μ1, μ2, cap=1000, excl_zero=false)
    # Marginal Q2
    marginal_q2 = Dict(k[2] => 0.0 for k in keys(emp_dist))
    for (k,v) in emp_dist
        marginal_q2[k[2]] += v
    end

    ρ = mean_arrival_rate(λ, μ1, θ) / μ2

    max_q2 = maximum(collect(keys(marginal_q2)))

    maxλ =  max_lambda(μ1, μ2, θ, 1)
    M = new_M_fn(maxλ, θ, μ1, μ2)

    exp_dist = Exponential(M)
    ks_stat = 0.0
    emp_surv_fn = 1.0-get(marginal_q2, 0, 0.0)
    if !excl_zero
        exp_surv_fn_left = 1.0 - cdf(exp_dist, 0.0)
        exp_surv_fn_right = 1.0 - cdf(exp_dist, (1-ρ)*1)

        ks_stat = max(ks_stat, abs(exp_surv_fn_left - emp_surv_fn))
        ks_stat = max(ks_stat, abs(exp_surv_fn_right - emp_surv_fn))
    end
    
    for k in 1:max_q2-1
        if (1-ρ)*k > cap
            break
        end
        emp_surv_fn -= get(marginal_q2, k, 0.0)

        exp_surv_fn_left = 1.0 - cdf(exp_dist, (1-ρ)*k)
        exp_surv_fn_right = 1.0 - cdf(exp_dist, (1-ρ)*(k+1))

        ks_stat = max(ks_stat, abs(exp_surv_fn_left - emp_surv_fn))
        ks_stat = max(ks_stat, abs(exp_surv_fn_right - emp_surv_fn))

    end 
    return ks_stat
end

#=μ1s = [1.1,1.2,1.5,2,4]
ρs = [.6, .7, .8, .9, .95, .99]
μ2 = 1
θs = [.1, .3, .5]

for μ1 in μ1s
    for θ in θs
        maxλ = max_lambda(μ1, μ2, θ, 1)

        M = M_fn(maxλ, θ, μ1, μ2)
        if M < 0
            println("Warning: M is negative for μ1: $μ1, μ2: $μ2, θ: $θ. M: $M")
            # throw(ErrorException("M is negative"))
        end

        for ρ in ρs
            println("Running simulation for μ1: $μ1, μ2: $μ2, θ: $θ, ρ: $ρ")
            println("Max λ for ρ=1: $maxλ, M: $M")
            λ = max_lambda(μ1, μ2, θ, ρ)
            empdist = emp_distn(λ, μ1, μ2, θ, 10000, 100000)

            # Mean Queue Length from Empirical and Heavy Traffic Approximation
            mql = sum(k[2]*v for (k,v) in empdist)
            HT_approx = M/(1-ρ)
            

            #Correlation between q1 and q2

            eq1 = eq2 = eq1q2 = eq1sq = eq2sq = 0.0

            for ((q1, q2), p) in empdist
                eq1  += q1 * p
                eq2  += q2 * p
                eq1q2 += q1 * q2 * p
                eq1sq += q1^2 * p
                eq2sq += q2^2 * p
            end

            cov_q1q2 = eq1q2 - eq1 * eq2
            var_q1  = eq1sq - eq1^2
            var_q2  = eq2sq - eq2^2
            corr   = cov_q1q2 / sqrt(var_q1 * var_q2)

            # Marginal Distribution of Q2
            
            marginal_q2 = Dict(k[2] => 0.0 for k in keys(empdist))
            for (k,v) in empdist
                marginal_q2[k[2]] += v
            end

            #Test Q2 Goodness of Fit to Exponential(1/M)
            #=exp_dist = Exponential(1/M)
            ks_stat = 0.0
            for (k,v) in marginal_q2
                ks_stat = max(ks_stat, abs(cdf(exp_dist, k) - sum(marginal_q2[j] for j in keys(marginal_q2) if j <= k)))
            end=#
            
            println("Mean Queue Length - Empirical: $mql, HT Approx: $HT_approx")
            println("Correlation between Q1 and Q2: $corr")
            # println("KS Statistic for Q2 vs Exponential(1/M): $ks_stat")
            println("----------")
            println()

        end
    end
end =#

sim_start_time = Dates.now()
# Finding Empirical Distributions for Q1-Q3

# Param Sets
μ1s = [1.1, 1.2, 1.5, 2, 4]
ρs = [.8, .9, .95, .99]
θs = [.1, .3, .5]
μ2 = 1.0

B = 100 # Number of batches to run per parameter set
N = round(Int, 10^8/ B)  # Number of jobs to simulate per batch

println("Starting empirical distribution simulations...")
println("B= $B, N= $N")
emp_distns = Dict{Tuple{Float64, Float64, Float64, Float64}, Dict{Tuple{Int64, Int64}, Float64}}()
batch_emp_distns = Dict{Tuple{Float64, Float64, Float64, Float64}, Vector{Dict{Tuple{Int64, Int64}, Float64}}}()
# ht_qls = Dict{Tuple{Float64, Float64, Float64, Float64}, Float64}()

for μ1 in μ1s
    for θ in θs
        maxλ = max_lambda(μ1, μ2, θ, 1)



        for ρ in ρs
            println("Running simulation for μ1: $μ1, μ2: $μ2, θ: $θ, ρ: $ρ")
            # println("Max λ for ρ=$ρ: $maxλ, M: $M")
            λ = max_lambda(μ1, μ2, θ, ρ)
            for b in 1:B
                # println("Starting batch $b of $B")
                batch_emp_distn = emp_distn(λ, μ1, μ2, θ, N, 10^8, false)
                if haskey(batch_emp_distns, (μ1, μ2, θ, ρ)) == false
                    batch_emp_distns[(μ1, μ2, θ, ρ)] = Vector{Dict{Tuple{Int64, Int64}, Float64}}()
                end
                push!(batch_emp_distns[(μ1, μ2, θ, ρ)], batch_emp_distn)
            end
            # Combine batch empirical distributions
            combined_emp_distn = Dict{Tuple{Int64, Int64}, Float64}()
            for batch_distn in batch_emp_distns[(μ1, μ2, θ, ρ)]
                for (k,v) in batch_distn
                    if haskey(combined_emp_distn, k) == false
                        combined_emp_distn[k] = 0.0
                    end
                    combined_emp_distn[k] += v / B
                end
            end
            #Normalize each batch empirical distribution
            for batch_distn in batch_emp_distns[(μ1, μ2, θ, ρ)]
                sm = sum(v for (k,v) in batch_distn)
                for (k,v) in batch_distn
                    batch_distn[k] = v / sm
                end
            end
            #Normalize the combined empirical distribution
            sm = sum(v for (k,v) in combined_emp_distn)
            for (k,v) in combined_emp_distn
                combined_emp_distn[k] = v / sm
            end

            emp_distns[(μ1, μ2, θ, ρ)] = combined_emp_distn

            # emp_dist = emp_distn(λ, μ1, μ2, θ, N, 10^7)                
            # emp_distns[(μ1, μ2, θ, ρ)] = emp_dist
            # empdist = emp_distn(λ, μ1, μ2, θ, N, 10^7)
            # emp_distns[(μ1, μ2, θ, ρ)] = empdist

            # Mean Queue Length from Empirical and Heavy Traffic Approximation
            #mql = sum(k[2]*v for (k,v) in empdist)
            #HT_approx = M/(1-ρ)

            #emp_qls[(μ1, μ2, θ, ρ)] = mql
            #ht_qls[(μ1, μ2, θ, ρ)] = HT_approx

            #println("Mean Queue Length - Empirical: $mql, HT Approx: $HT_approx")
            #println("----------")
            #println()

        end
    end
end
println("Completed all empirical distributions.")
cur_time = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")

# Save results to file for later analysis
serialize(string(raw"C:\Users\ccurt\Documents\params", cur_time, ".jls"), (B, N, μ1s, μ2, θs, ρs))
serialize(string(raw"C:\Users\ccurt\Documents\emp_distns_", cur_time, ".jls"), emp_distns)
serialize(string(raw"C:\Users\ccurt\Documents\batch_emp_distns_", cur_time, ".jls"), batch_emp_distns)
println("Results written to CSV files.")
println("Total Simulation Time: ", Dates.now() - sim_start_time)

# Empirical Distributions For Q4


# Compute KS Statistics
#=
println("Starting KS Statistic Calculations...")
emp_distns = deserialize(raw"C:\Users\ccurt\Documents\emp_distns_2026-01-19_233317.jls")
batch_emp_distns = deserialize(raw"C:\Users\ccurt\Documents\batch_emp_distns_2026-01-19_233317.jls")

ks_stats = DataFrame(mu1=Float64[], mu2=Float64[], theta=Float64[], rho=Float64[], MKS_Statistic=Float64[], KS_Statistic=Float64[], KS_Statistic_Excl_Zero=Float64[]) 
for ((μ1, μ2, θ, ρ), empdist) in emp_distns
    mks_stat = compute_MKS(empdist, max_lambda(μ1, μ2, θ, ρ), θ, μ1, μ2, 5)
    ks_stat = compute_KS(empdist, max_lambda(μ1, μ2, θ, ρ), θ, μ1, μ2)
    ks_stat_excl_zero = compute_KS(empdist, max_lambda(μ1, μ2, θ, ρ), θ, μ1, μ2, 1000, true)
    push!(ks_stats, (μ1, μ2, θ, ρ, mks_stat, ks_stat, ks_stat_excl_zero))
end
println("Completed KS Statistic Calculations.")
cur_time = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
CSV.write(string(raw"C:\Users\ccurt\Documents\ks_stats_", cur_time, ".csv"), ks_stats)=#

# Plotting Empirical CDFs of (1-ρ) Q2
#=for μ1 in μ1s
    for θ in θs
        p = plot(title = "Empirical CDFs of (1-ρ) Q₂ for μ₁=$(μ1), θ=$(θ)",
            xlabel = L"(1-\rho) Q_2",
            ylabel = "Empirical CDF",
            xlim = (0, 5),
            ylim = (0, 1),
            legend = :bottomright,
            size = (1000, 750)
        )
        for ρ in ρs
            empdist = emp_distns[(μ1, μ2, θ, ρ)]
            # Display CDF of (1-ρ) Q_2
            marginal_q2s = Dict(k[2] => 0.0 for k in keys(empdist))
            for (k,v) in empdist
                marginal_q2s[k[2]] += v
            end
            max_q2 = maximum(collect(keys(marginal_q2s)))
            
            emp_cdf = [sum(marginal_q2s[j] for j in keys(marginal_q2s) if j <= k) for k in 0:max_q2]
            
            ρ = mean_arrival_rate(max_lambda(μ1, μ2, θ, ρ), μ1, θ) / μ2

            xs = zeros(max_q2 + 1)
            ys = zeros(max_q2 + 1)

            for k in 0:max_q2
                xs[k+1] = (1-ρ) * k
                ys[k+1] = emp_cdf[k+1]
            end

            scatter!(p, xs, ys, label = "ρ=$ρ", markersize=3)

        end
        cur_time = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
        # Theoretical CDF
        M = new_M_fn(max_lambda(μ1, μ2, θ, 1), θ, μ1, μ2)
        xs_theory = 0:0.01:5
        ys_theory = cdf.(Exponential(M), xs_theory)
        plot!(p, xs_theory, ys_theory, label = "Theoretical CDF", linewidth=2, linecolor=:black)
        savefig(p, string("C:\\Users\\ccurt\\Documents\\Plots\\", cur_time, "_mu1_", μ1, "_theta_", θ, ".png"))
    end
end=#

# Question 1: Mean Queue Length Comparison
#=mean_queue_lengths = DataFrame(mu1=Float64[], mu2=Float64[], theta=Float64[], rho=Float64[], Empirical_Mean_Q2=Float64[], HT_Approx_Mean_Q2=Float64[], Uncorrected_HT_Approx_Mean_Q2=Float64[], Empirical_Mean_Q2_5th_Percentile=Float64[], Empirical_Mean_Q2_95th_Percentile=Float64[])
for ((μ1, μ2, θ, ρ), empdist) in emp_distns
    # Batch Stats
    batch_mqls = Float64[]
    for b in 1:B
        batch_empdist = batch_emp_distns[(μ1, μ2, θ, ρ)][b]
        batch_emp_queue2 = sum(k[2]*v for (k,v) in batch_empdist)
        # push!(mean_queue_lengths, (μ1, μ2, θ, ρ, batch_emp_queue2, NaN, NaN))
        push!(batch_mqls, batch_emp_queue2)
    end
    emp_queue2_5th_percentile = quantile(batch_mqls, 0.05)
    emp_queue2_95th_percentile = quantile(batch_mqls, 0.95)
    local emp_queue2 = sum(k[2]*v for (k,v) in empdist)
    local M = new_M_fn(max_lambda(μ1, μ2, θ, 1), θ, μ1, μ2)
    local uncorr_M = new_M_fn(max_lambda(μ1, μ2, θ, ρ), θ, μ1, μ2)
    if M < 0
        println("Warning: M is negative for ρ: $ρ, μ1: $μ1, μ2: $μ2, θ: $θ. M: $M")
        throw(ErrorException("M is negative"))
    end
    HT_approx_queue2 = M/(1-ρ)
    uncorr_HT_approx_queue2 = uncorr_M/(1-ρ)
    push!(mean_queue_lengths, (μ1, μ2, θ, ρ, emp_queue2, HT_approx_queue2, uncorr_HT_approx_queue2, emp_queue2_5th_percentile, emp_queue2_95th_percentile))
end
println("Completed Mean Queue Length Comparison.")
# Question 2: Check for correlation between Q1 and Q2

corrs = DataFrame(mu1=Float64[], mu2=Float64[], theta=Float64[], rho=Float64[], Correlation_Q1_Q2=Float64[], Correlation_5th_Percentile=Float64[], Correlation_95th_Percentile=Float64[])
for ((μ1, μ2, θ, ρ), empdist) in emp_distns
    #Batch Stats
    batch_corrs = Float64[]
    for b in 1:B
        batch_empdist = batch_emp_distns[(μ1, μ2, θ, ρ)][b]
        eq1 = eq2 = eq1q2 = eq1sq = eq2sq = 0.0

        for ((q1, q2), p) in batch_empdist
            eq1  += q1 * p
            eq2  += q2 * p
            eq1q2 += q1 * q2 * p
            eq1sq += q1^2 * p
            eq2sq += q2^2 * p
        end

        cov_q1q2 = eq1q2 - eq1 * eq2
        var_q1  = eq1sq - eq1^2
        var_q2  = eq2sq - eq2^2
        corr   = cov_q1q2 / sqrt(var_q1 * var_q2)
        push!(batch_corrs, corr)
    end

    emp_corr_5th_percentile = quantile(batch_corrs, 0.05)
    emp_corr_95th_percentile = quantile(batch_corrs, 0.95)

    eq1 = eq2 = eq1q2 = eq1sq = eq2sq = 0.0

    for ((q1, q2), p) in empdist
        eq1  += q1 * p
        eq2  += q2 * p
        eq1q2 += q1 * q2 * p
        eq1sq += q1^2 * p
        eq2sq += q2^2 * p
    end

    cov_q1q2 = eq1q2 - eq1 * eq2
    var_q1  = eq1sq - eq1^2
    var_q2  = eq2sq - eq2^2
    corr   = cov_q1q2 / sqrt(var_q1 * var_q2)

    push!(corrs, (μ1, μ2, θ, ρ, corr, emp_corr_5th_percentile, emp_corr_95th_percentile))
end
println("Completed Correlation Calculations.")=#
# Question 3: Marginal Distribution of Q2 and Goodness of Fit to Exponential
#= gof_q2 = DataFrame(mu1=Float64[], mu2=Float64[], theta=Float64[], rho=Float64[], KS_Statistic=Float64[], uncorrected_KS_Statistic=Float64[], KS_5th_Percentile=Float64[], KS_95th_Percentile=Float64[], uncorrected_KS_5th_Percentile=Float64[], uncorrected_KS_95th_Percentile=Float64[])
for ((μ1, μ2, θ, ρ), empdist) in emp_distns
    marginal_q2s = Dict(k[2] => 0.0 for k in keys(empdist))

    for (k,v) in empdist
        marginal_q2s[k[2]] += v
    end

    max_q2 = maximum(collect(keys(marginal_q2s)))

    local M = new_M_fn(max_lambda(μ1, μ2, θ, 1), θ, μ1, μ2)
    local uncorrected_M = new_M_fn(max_lambda(μ1, μ2, θ, ρ), θ, μ1, μ2)

    exp_dist = Exponential(1/M)
    uncorrected_exp_dist = Exponential(1/uncorrected_M)
    # Batch Stats
    batch_ks_stats = Float64[]
    batch_uncorrected_ks_stats = Float64[]
    for b in 1:B
        batch_empdist = batch_emp_distns[(μ1, μ2, θ, ρ)][b]
        marginal_q2s = Dict(k[2] => 0.0 for k in keys(batch_empdist))

        for (k,v) in batch_empdist
            marginal_q2s[k[2]] += v
        end

        cur_k = 1
        emp_cdf = marginal_q2s[0]
        ks_stat = emp_cdf
        uncorrected_ks_stat = emp_cdf
        while cur_k <= max_q2
            left_emp = emp_cdf
            right_emp = emp_cdf + get(marginal_q2s, cur_k, 0.0)

            exp_cdf = cdf(exp_dist, (1-ρ)*cur_k)
            uncorrected_exp_cdf = cdf(uncorrected_exp_dist, (1-ρ)*cur_k)
            
            ks_stat = max(ks_stat, abs(exp_cdf - left_emp))
            ks_stat = max(ks_stat, abs(exp_cdf - right_emp))

            uncorrected_ks_stat = max(uncorrected_ks_stat, abs(uncorrected_exp_cdf - left_emp))
            uncorrected_ks_stat = max(uncorrected_ks_stat, abs(uncorrected_exp_cdf - right_emp))

            emp_cdf += get(marginal_q2s, cur_k, 0.0)
            cur_k += 1
        end
        push!(batch_ks_stats, ks_stat)
        push!(batch_uncorrected_ks_stats, uncorrected_ks_stat)
    end
    ks_stat_5th_percentile = quantile(batch_ks_stats, 0.05)
    ks_stat_95th_percentile = quantile(batch_ks_stats, 0.95)
    uncorrected_ks_stat_5th_percentile = quantile(batch_uncorrected_ks_stats, 0.05)
    uncorrected_ks_stat_95th_percentile = quantile(batch_uncorrected_ks_stats, 0.95)
    # Combined Stats
    cur_k = 1
    emp_cdf = marginal_q2s[0]
    ks_stat = emp_cdf
    uncorrected_ks_stat = emp_cdf
    while cur_k <= max_q2
        left_emp = emp_cdf
        right_emp = emp_cdf + get(marginal_q2s, cur_k, 0.0)

        exp_cdf = cdf(exp_dist, (1-ρ)*cur_k)
        uncorrected_exp_cdf = cdf(uncorrected_exp_dist, (1-ρ)*cur_k)
        
        ks_stat = max(ks_stat, abs(exp_cdf - left_emp))
        ks_stat = max(ks_stat, abs(exp_cdf - right_emp))

        uncorrected_ks_stat = max(uncorrected_ks_stat, abs(uncorrected_exp_cdf - left_emp))
        uncorrected_ks_stat = max(uncorrected_ks_stat, abs(uncorrected_exp_cdf - right_emp))

        emp_cdf += get(marginal_q2s, cur_k, 0.0)
        cur_k += 1
    end

    
    #emp_cdf = [sum(marginal_q2s[j] for j in keys(marginal_q2s) if j <= k) for k in 0:max_q2]
    # scaled_q2s = (1-ρ) .* marginal_q2s

    #M = new_M_fn(max_lambda(μ1, μ2, θ, 1), θ, μ1, μ2)

    # Mn = (1-ρ) * sum(k*v for (k,v) in marginal_q2)
    # exp_dist = Exponential(1/M)
    

    push!(gof_q2, (μ1, μ2, θ, ρ, ks_stat, uncorrected_ks_stat, ks_stat_5th_percentile, ks_stat_95th_percentile, uncorrected_ks_stat_5th_percentile, uncorrected_ks_stat_95th_percentile))
end
println("Completed Goodness of Fit Calculations.")

# Write Results to CSV
cur_time = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")

CSV.write(string(raw"C:\Users\ccurt\Documents\mean_queue_lengths", cur_time, ".csv"), mean_queue_lengths)
CSV.write(string(raw"C:\Users\ccurt\Documents\correlations_q1_q2", cur_time, ".csv"), corrs)
CSV.write(string(raw"C:\Users\ccurt\Documents\gof_q2", cur_time,".csv"), gof_q2)

println("Results written to CSV files.")
println("Total Simulation Time: ", Dates.now() - sim_start_time)=#


# Question 4: Optimal Capacity Planning
#=capacity_plans = DataFrame(mu1=Float64[], theta=Float64[], lam=Float64[], Optimal_Mu2=Float64[], HT_Optimal_Mu2=Float64[], Empirical_Cost=Float64[], Empirical_Cost_5th=Float64[], Empirical_Cost_95th=Float64[], HT_Cost=Float64[], Approx_Error=Float64[], Approx_Error_5th=Float64[], Approx_Error_95th=Float64[])

μ1s = [1.1, 1.2, 1.5, 2, 4]
λs = [0.5, 0.7, 0.9, 1.0, 1.2]
θs = [.1, .3, .5]

wait_cost = 1.0
capacity_cost = 2.0

for μ1 in μ1s
    for θ in θs
        # max_λ = max_lambda(μ1, μ2, θ, 1)
        # M = new_M_fn(max_λ, θ, μ1, μ2)
        for λ in λs
            local M = new_M_fn(λ, θ, μ1, μ2)
            println("Running capacity planning for μ1: $μ1, θ: $θ, λ: $λ")
            min_μ2 = mean_arrival_rate(λ, μ1, θ)
            max_μ2 = 2 * min_μ2
            
            # ht_opt_μ2 = wait_cost + sqrt(wait_cost * M / capacity_cost)

            μ2_vals = collect(range(min_μ2 + 0.01, max_μ2, length=100))
            emp_costs = zeros(length(μ2_vals))
            batch_emp_costs = zeros(length(μ2_vals), B)
            ht_costs = zeros(length(μ2_vals))

            for (i, μ2) in enumerate(μ2_vals)
                # Generate B batches of empirical distributions
                batch_emp_distns = Vector{Dict{Tuple{Int64, Int64}, Float64}}()
                for b in 1:B
                    batch_emp_distn = emp_distn(λ, μ1, μ2, θ, N, 10^7)
                    push!(batch_emp_distns, batch_emp_distn)
                    batch_emp_q2 = sum(k[2]*v for (k,v) in batch_emp_distn)
                    batch_emp_costs[i, b] = wait_cost * batch_emp_q2 + capacity_cost * μ2
                end
                # Combine batches into single empirical distribution
                combined_emp_dist = Dict{Tuple{Int64, Int64}, Float64}()
                for batch_dist in batch_emp_distns
                    for (k,v) in batch_dist
                        if haskey(combined_emp_dist, k) == false
                            combined_emp_dist[k] = 0.0
                        end
                        combined_emp_dist[k] += v / B
                    end
                end

                emp_dist = combined_emp_dist
                emp_queue2 = sum(k[2]*v for (k,v) in emp_dist)
                emp_cost = wait_cost * emp_queue2 + capacity_cost * μ2

                emp_costs[i] = emp_cost
                ht_costs[i] = wait_cost * (M / (1-min_μ2/μ2)) + capacity_cost * μ2
            end
            # Optimal under HT Approximation
            ht_opt_μ2_idx = argmin(ht_costs)
            ht_opt_cost = ht_costs[ht_opt_μ2_idx]
            ht_opt_μ2 = μ2_vals[ht_opt_μ2_idx]

            # Batch Stats for Empirical Costs
            batch_opt_costs = Float64[]
            batch_approx_errors = Float64[]
            for b in 1:B
                batch_emp_opt_μ2_idx = argmin(batch_emp_costs[:, b])
                batch_emp_opt_cost = batch_emp_costs[batch_emp_opt_μ2_idx, b]
                batch_emp_opt_μ2 = μ2_vals[batch_emp_opt_μ2_idx]

                batch_approx_error = emp_costs[ht_opt_μ2_idx] - emp_costs[batch_emp_opt_μ2_idx]
                push!(batch_opt_costs, batch_emp_opt_cost)
                push!(batch_approx_errors, batch_approx_error)
            end
            opt_cost_5th_percentile = quantile(batch_opt_costs, 0.05)
            opt_cost_95th_percentile = quantile(batch_opt_costs, 0.95)
            approx_error_5th_percentile = quantile(batch_approx_errors, 0.05)
            approx_error_95th_percentile = quantile(batch_approx_errors, 0.95)
            
            # Overall Stats for Empirical Costs

            emp_opt_μ2_idx = argmin(emp_costs)
            emp_opt_cost = emp_costs[emp_opt_μ2_idx]
            emp_opt_μ2 = μ2_vals[emp_opt_μ2_idx]

            approx_error = emp_costs[ht_opt_μ2_idx] - emp_costs[emp_opt_μ2_idx]

            # emp_dist = emp_distn(λ, μ1, min_μ2, θ, N, 10^7)


            push!(capacity_plans, (μ1, θ, λ, emp_opt_μ2, ht_opt_μ2, emp_opt_cost, opt_cost_5th_percentile, opt_cost_95th_percentile, ht_opt_cost, approx_error, approx_error_5th_percentile, approx_error_95th_percentile))
        end
    end
end

CSV.write(string(raw"C:\Users\ccurt\Documents\staticdecision", cur_time,".csv"), capacity_plans)

println("Total Simulation Time: ", Dates.now() - sim_start_time)
println("N: $N")
=#
