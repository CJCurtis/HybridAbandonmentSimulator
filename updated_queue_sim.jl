using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Plots")
Pkg.add("Distributions")
Pkg.add("LaTeXStrings")
Pkg.add("Random")
Pkg.add("StatsBase")
Pkg.add("CSV")
Pkg.add("HypothesisTests")
Pkg.add("Dates")
Pkg.add("Serialization")
using Random, Distributions, Plots, LaTeXStrings, CSV, DataFrames, StatsBase, CSV, HypothesisTests, Dates, Serialization 

random_seed = 3292001

function emp_distn(λ, μ1, μ2, θ, max_jobs=10^8, update_every=10^9, norm=true)
    pd_num = 0
    jobs_completed = 0


    states = Dict{Tuple{Int,Int}, Float64}()
    init_time = time()
    t = 0.0
    q1 = 1
    q2 = 0
    while jobs_completed < max_jobs
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


        u = rand()
        if u < arr_rate/rate
            # Arrival

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

function mean_arrival_rate(λ, μ1, θ, tol=1e-8)
    pisdivp0 = ones(1000)
    pisdivp0[1] = λ/(μ1)
    for i in 2:1000
        pisdivp0[i] = pisdivp0[i-1]*(λ) / (μ1 + θ * (i-1) )
    end
    pi0 = 1/(1 + sum(pisdivp0))
    return (1-pi0) * μ1
end

function max_lambda(μ1, μ2, θ, ρ=1, tolerance=1e-6)
    lam = 0
    step = 1
    dif =  (ρ - mean_arrival_rate(lam, μ1, θ)/μ2)
    while abs(dif) > tolerance
        new_lam = lam + step * sign(dif)
        new_dif =  (ρ - mean_arrival_rate(new_lam, μ1, θ)/μ2)
        if abs(new_dif) >= abs(dif)
            step /= 2
        else
            lam = new_lam
            dif = new_dif
        end
    end 
    return lam
end


function new_M_fn(λ, θ, μ1, μ2)
    n_states = 1000
    pis = ones(1, n_states)
    for i in 1:(n_states-1)
        pis[1, i+1] = pis[1, i] * λ/(μ1 + θ * (i-1))
    end
    pis = pis ./ sum(pis)

    imf = zeros(n_states, n_states)
    for i in 1:n_states
        imf[i, i] = μ2
        if i > 1
            imf[i, i-1] = -μ1
        end
    end

    P = zeros(n_states, n_states)
    P[1, 1] = -λ
    P[1, 2] = λ
    for i in 2:(n_states-1)
        P[i, i+1] = λ
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

function mean_ql(λ, θ, μ1, μ2, num_jobs=10^8)
    q1 = 0
    q2 = 0
    total_q2 = 0.0
    num_observations = 0
    while num_observations < num_jobs
        arr_rate = λ
        serv1_rate = q1 > 0 ? μ1 : 0
        serv2_rate = q2 > 0 ? μ2 : 0
        aband_rate = q1 > 0 ? (q1-1)*θ : 0
        rate = arr_rate + serv1_rate + serv2_rate + aband_rate
        u = rand()
        if u < arr_rate/rate
            # Arrival
            q1 += 1
            total_q2 += q2
            num_observations += 1
        elseif u < (arr_rate + serv1_rate)/rate
            # Service completion at server 1
            q1 -= 1
            q2 += 1
        elseif u < (arr_rate + serv1_rate + serv2_rate)/rate
            # Service completion at server 2
            q2 -= 1
        else
            # Abandonment from queue 1
            q1 -= 1
        end
        
    end
    return total_q2 / num_observations
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

function compute_KS(emp_dist, λ, θ, μ1, μ2, excl_zero=false)
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

        emp_surv_fn -= get(marginal_q2, k, 0.0)

        exp_surv_fn_left = 1.0 - cdf(exp_dist, (1-ρ)*k)
        exp_surv_fn_right = 1.0 - cdf(exp_dist, (1-ρ)*(k+1))

        ks_stat = max(ks_stat, abs(exp_surv_fn_left - emp_surv_fn))
        ks_stat = max(ks_stat, abs(exp_surv_fn_right - emp_surv_fn))

    end 
    return ks_stat
end

function compute_KS_Geo(emp_dist, λ, θ, μ1, μ2)
    # Marginal Q2
    marginal_q2 = Dict(k[2] => 0.0 for k in keys(emp_dist))
    for (k,v) in emp_dist
        marginal_q2[k[2]] += v
    end

    ρ = mean_arrival_rate(λ, μ1, θ) / μ2

    max_q2 = maximum(collect(keys(marginal_q2)))

    M = new_M_fn(max_lambda(μ1, μ2, θ, 1), θ, μ1, μ2)

    mean_ql_approx = M / (1 - ρ)
    p = 1 / (1 + mean_ql_approx)

    ks_stat = 0.0
    emp_surv_fn = 1.0

    
    for k in 0:max_q2-1


        geo_surv_fn = (1-p)^k

        ks_stat = max(ks_stat, abs(geo_surv_fn - emp_surv_fn))

        emp_surv_fn -= get(marginal_q2, k, 0.0)


    end 
    return ks_stat
end

function genrQ1Q3_emp_distns(μ1s, μ2, θs, ρs, B, N)
    emp_distns = Dict{Tuple{Float64, Float64, Float64, Float64}, Dict{Tuple{Int64, Int64}, Float64}}()
    batch_emp_distns = Dict{Tuple{Float64, Float64, Float64, Float64}, Vector{Dict{Tuple{Int64, Int64}, Float64}}}()
    num_settings = length(μ1s) * length(θs) * length(ρs)
    start_time = Dates.now()
    setting_count = 0
    Random.seed!(random_seed)
    for μ1 in μ1s
        for θ in θs
            for ρ in ρs
                setting_count += 1
                λ = max_lambda(μ1, μ2, θ, ρ)
                println("Simulating for parameters: μ1=$μ1, μ2=$μ2, θ=$θ, ρ=$ρ, λ=$λ")
                batch_dists = Vector{Dict{Tuple{Int64, Int64}, Float64}}()
                for b in 1:B
                    emp_dist = emp_distn(λ, μ1, μ2, θ, N, )
                    push!(batch_dists, emp_dist)
                end
                # Combine batch distributions
                combined_dist = Dict{Tuple{Int64, Int64}, Float64}()
                for dist in batch_dists
                    for (k,v) in dist
                        if haskey(combined_dist, k) == false
                            combined_dist[k] = 0.0
                        end
                        combined_dist[k] += v / B
                    end
                end
                emp_distns[(μ1, μ2, θ, ρ)] = combined_dist
                batch_emp_distns[(μ1, μ2, θ, ρ)] = batch_dists
                println("Completed $setting_count out of $num_settings settings. $((setting_count/num_settings)*100) %")
                println("Elapsed time: ", Dates.now() - start_time)
                println("----------")
            end
        end
    end
    return emp_distns, batch_emp_distns
end

function genrQ4_mqls(μ1s, θs, λs, num_μ2s, B, N)
    emp_mqls = Dict{Tuple{Float64, Float64, Float64}, Dict{Float64, Float64}}()
    batch_mqls = Dict{Tuple{Float64, Float64, Float64}, Dict{Float64, Vector{Float64}}}()
    num_runs = length(μ1s) * length(θs) * length(λs) * num_μ2s
    run_count = 0
    Random.seed!(random_seed)
    for μ1 in μ1s
        for θ in θs
            for λ in λs
                λeff = mean_arrival_rate(λ, μ1, θ)
                M0 = new_M_fn(λ, θ, μ1, λeff)

                opt_mu_guess_low = λeff + sqrt(λeff * M0/2)
                opt_mu_guess_high = λeff + sqrt(λeff * M0/.5)

                μ2s = range(λeff/.7, stop=4 * λeff, length=num_μ2s)
                λeff = mean_arrival_rate(λ, μ1, θ)
                println("Simulating for parameters: μ1=$μ1, θ=$θ, λ=$λ")
                μ2_to_mql = Dict{Float64, Float64}()
                μ2_to_batch_mqls = Dict{Float64, Vector{Float64}}()
                for μ2 in μ2s
                    run_count += 1
                    ρ = λeff / μ2
                    # println("μ2: $μ2, ρ: $ρ")
                    batch_mql = Vector{Float64}()
                    for b in 1:B
                        emp_dist = emp_distn(λ, μ1, μ2, θ, N)
                        emp_queue2 = sum(k[2]*v for (k,v) in emp_dist)

                        push!(batch_mql, emp_queue2)
                    end
                    # Combine batch distributions
                    mql = mean(batch_mql)
                    μ2_to_mql[μ2] = mql

                    μ2_to_batch_mqls[μ2] = batch_mql
                    
                    emp_mqls[(λ, θ, μ1)] = μ2_to_mql
                    batch_mqls[(λ, θ, μ1)] = μ2_to_batch_mqls

                end

                println("Completed $run_count out of $num_runs runs. $((run_count/num_runs)*100) %")
                
            end
        end
    end
    return emp_mqls, batch_mqls
end

function mean_ql_comparison(emp_distns, batch_emp_distns, B)
    mean_queue_lengths = DataFrame(mu1=Float64[], mu2=Float64[], theta=Float64[], rho=Float64[], Empirical_Mean_Q2=Float64[], Empirical_Mean_Q2_Std=Float64[], M0 = Float64[], ScaledEQ2=Float64[], Empirical_Rho=Float64[], Empirical_Rho_Std=Float64[])
    num_settings = length(emp_distns)
    setting_count = 0
    start_time = Dates.now()
    for ((μ1, μ2, θ, ρ), empdist) in emp_distns
        # Batch Stats
        batch_mqls = Float64[]
        batch_rhos = Float64[]
        for b in 1:B
            batch_empdist = batch_emp_distns[(μ1, μ2, θ, ρ)][b]
            batch_emp_queue2 = sum(k[2]*v for (k,v) in batch_empdist)
            push!(batch_mqls, batch_emp_queue2)

            batch_emp_rho = sum(v for (k,v) in batch_empdist if k[2] > 0)
            push!(batch_rhos, batch_emp_rho)
        end
        emp_queue2_std = std(batch_mqls)

        emp_rho_std = std(batch_rhos)

        local emp_queue2 = sum(k[2]*v for (k,v) in empdist)
        emp_rho = sum(v for (k,v) in empdist if k[2] > 0)
        local M = new_M_fn(max_lambda(μ1, μ2, θ, 1), θ, μ1, μ2)
        if M < 0
            println("Warning: M is negative for ρ: $ρ, μ1: $μ1, μ2: $μ2, θ: $θ. M: $M")
            throw(ErrorException("M is negative"))
        end
        push!(mean_queue_lengths, (μ1, μ2, θ, ρ, emp_queue2, emp_queue2_std, M, emp_queue2 * (1-ρ), emp_rho, emp_rho_std))
        setting_count += 1
        println("Completed $setting_count out of $num_settings settings. $((setting_count/num_settings)*100) %")
        println("Elapsed time: ", Dates.now() - start_time)
        println("----------")
    end
    return mean_queue_lengths
end


function corr_comparison(emp_distns, batch_emp_distns, B)
    corrs = DataFrame(mu1=Float64[], mu2=Float64[], theta=Float64[], rho=Float64[], Correlation_Q1_Q2=Float64[], Correlation_Std=Float64[])
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

        emp_corr_std = std(batch_corrs)

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

        push!(corrs, (μ1, μ2, θ, ρ, corr, emp_corr_std))
    end
    return corrs
end

function KS_comparison(emp_distns, batch_emp_distns, B)
    ks_stats = DataFrame(mu1=Float64[], mu2=Float64[], theta=Float64[], rho=Float64[], KS_Statistic=Float64[], KS_Std=Float64[], KS_Statistic_Geo=Float64[], KS_Statistic_Geo_Std=Float64[])
    num_settings = length(emp_distns)
    setting_count = 0
    start_time = Dates.now()
    for ((μ1, μ2, θ, ρ), empdist) in emp_distns
        # Batch Stats
        batch_ks_stats = Float64[]
        batch_geo_ks_stats = Float64[]
        for b in 1:B
            batch_empdist = batch_emp_distns[(μ1, μ2, θ, ρ)][b]
            ks_stat = compute_KS(batch_empdist, max_lambda(μ1, μ2, θ, ρ), θ, μ1, μ2)
            geo_ks_stat = compute_KS_Geo(batch_empdist, max_lambda(μ1, μ2, θ, ρ), θ, μ1, μ2)
            push!(batch_ks_stats, ks_stat)
            push!(batch_geo_ks_stats, geo_ks_stat)
        end
        emp_ks_std = std(batch_ks_stats)
        emp_geo_ks_std = std(batch_geo_ks_stats)
        local emp_ks_stat = compute_KS(empdist, max_lambda(μ1, μ2, θ, ρ), θ, μ1, μ2)
        local emp_geo_ks_stat = compute_KS_Geo(empdist, max_lambda(μ1, μ2, θ, ρ), θ, μ1, μ2)
        push!(ks_stats, (μ1, μ2, θ, ρ, emp_ks_stat, emp_geo_ks_std, emp_geo_ks_stat, emp_geo_ks_std))
        setting_count += 1
        println("Completed $setting_count out of $num_settings settings. $((setting_count/num_settings)*100) %")
        println("Elapsed time: ", Dates.now() - start_time)
        println("----------")
    end
    return ks_stats
end

function opt_comparison(emp_mqls, batch_emp_mqls, B, cs=[.5, .75, 1.0, 1.5, 2.0])
    opt_costs = DataFrame(lambda=Float64[], mu1=Float64[], theta=Float64[],  c=Float64[], Empirical_Optimal_Cost=Float64[], Empirical_Optimal_Cost_Std=Float64[], Approx_Optimal_Cost=Float64[], Empirical_Optimal_Mu2=Float64[], Approx_Optimal_Mu2=Float64[], Empirical_Optimal_Rho=Float64[], Approx_Optimal_Rho=Float64[], Mismatch_cost = Float64[])
    for ((λ, θ, μ1), μ2_to_mql) in emp_mqls
        λeff = mean_arrival_rate(λ, μ1, θ)
        M0 = new_M_fn(λ, θ, μ1, λeff)
        emp_min_cost = Dict{Float64, Float64}() # Match c to min cost
        emp_min_μ2 = Dict{Float64, Float64}() # Match c to min cost's μ2
        batch_min_costs = Dict{Float64, Vector{Float64}}()
        approx_min_μ2 = Dict{Float64, Float64}()
        for c in cs
            # Initialize batch min costs for all cs as Inf, Inf, ...
            batch_min_costs[c] = Inf .* ones(B)
        end

        approx_min_costs = Dict{Float64, Float64}()

        for (μ2, emp_mql) in μ2_to_mql
            ρ = λeff / μ2
            approx_mql = M0 / (1 - ρ)
            
            

            for c in cs
                approx_cost = μ2 + c * approx_mql
                emp_cost = μ2 + c * emp_mql

                emp_min_cost[c] = min(get(emp_min_cost, c, Inf), emp_cost)
                if emp_min_cost[c] == emp_cost
                    emp_min_μ2[c] = μ2
                end

                approx_min_costs[c] = min(get(approx_min_costs, c, Inf), approx_cost)
                if approx_min_costs[c] == approx_cost
                    approx_min_μ2[c] = μ2
                end
                

                for b in 1:B
                    # Batch Stats
                    batch_emp_mql = batch_emp_mqls[(λ, θ, μ1)][μ2][b]
                    batch_emp_cost = μ2 + c * batch_emp_mql

                    batch_min_costs[c][b] = min(batch_min_costs[c][b], batch_emp_cost)
                end
            end
            

        end
        for c in cs
                approx_opt_μ2 = approx_min_μ2[c]
                emp_cost_at_approx_μ2 = approx_opt_μ2 + c * μ2_to_mql[approx_opt_μ2]

                mismatch = emp_min_cost[c] - emp_cost_at_approx_μ2

                batch_std = std(batch_min_costs[c])
                push!(opt_costs, (λ, μ1, θ, c, emp_min_cost[c],  batch_std, approx_min_costs[c], emp_min_μ2[c], approx_min_μ2[c], λeff / emp_min_μ2[c], λeff / approx_min_μ2[c], mismatch))
            end
        
        
        
    end
    return opt_costs
    
end



function cost_estimation(μ1s, θs, λs, cs, B, N, num_μ2s)
    opt_cost_results = DataFrame(lambda=Float64[], mu1=Float64[], theta=Float64[],  c=Float64[], Empirical_Optimal_Cost=Float64[], Empirical_Optimal_Cost_Std=Float64[], Approx_Optimal_Cost=Float64[], Empirical_Optimal_Mu2=Float64[], Approx_Optimal_Mu2=Float64[], Empirical_Optimal_Rho=Float64[], Approx_Optimal_Rho=Float64[], Mismatch_cost = Float64[])
    max_c = maximum(cs)
    min_c = minimum(cs)
    num_iters = length(μ1s) * length(θs) * length(λs)
    iter_count = 0
    start_time = Dates.now()
    for μ1 in μ1s
        for θ in θs
            for λ in λs
                iter_count += 1
                println("Starting iteration $iter_count out of $num_iters. Parameters: μ1=$μ1, θ=$θ, λ=$λ ")
                println("Elapsed time: ", Dates.now() - start_time)
                println("Percent complete: ", round(((iter_count-1)/num_iters)*100, digits=2), "%")
                λeff = mean_arrival_rate(λ, μ1, θ)
                M0 = new_M_fn(λ, θ, μ1, λeff)



                opt_mu_guess_low = λeff + sqrt(λeff * M0* min_c)
                opt_mu_guess_high = λeff + sqrt(λeff * M0*max_c)
                μ2s = range(.9 * opt_mu_guess_low, stop=1.1 * opt_mu_guess_high, length=num_μ2s)

                aggr_emp_mqls = zeros(Float64, num_μ2s)
                emp_min_costs = zeros(Float64, length(cs), B) # Match c to all B min costs
                for b in 1:B
                    Random.seed!(random_seed + b)
                    
                    approx_mqls = zeros(Float64, num_μ2s)
                    emp_mqls = zeros(Float64, num_μ2s)

                    for idx in 1:length(μ2s)
                        μ2 = μ2s[idx]
                        ρ = λeff / μ2
                        approx_mql = M0 / (1 - ρ)
                        
                        emp_queue2 = mean_ql(λ, θ, μ1, μ2, N)

                        aggr_emp_mqls[idx] += emp_queue2 / B

                        approx_mqls[idx] = approx_mql
                        emp_mqls[idx] = emp_queue2

                    end
                    for c_idx in 1:length(cs)
                        c = cs[c_idx]

                        emp_costs = μ2s .+ c .* emp_mqls
                        min_emp_cost, min_emp_idx = findmin(emp_costs)
                        emp_min_costs[c_idx, b] = min_emp_cost
                    end



                end

                for c_idx in 1:length(cs)
                    c = cs[c_idx]

                    emp_cost_std = std(emp_min_costs[c_idx, :])/sqrt(B)
                    approx_mqls = [M0 / (1 - (λeff / μ2)) for μ2 in μ2s]

                    approx_opt_cost, approx_min_idx = findmin(μ2s .+ c .* approx_mqls)
                    approx_opt_μ2 = μ2s[approx_min_idx]
                     

                    mismatched_cost = approx_opt_μ2 + c * aggr_emp_mqls[approx_min_idx]

                    emp_opt_cost, emp_min_idx = findmin(μ2s .+ c .* aggr_emp_mqls)
                    emp_opt_μ2 = μ2s[emp_min_idx]

                    emp_opt_rho = λeff / emp_opt_μ2
                    approx_opt_rho = λeff / approx_opt_μ2

                    mismatch = emp_opt_cost - mismatched_cost
                    

                    push!(opt_cost_results, (λ, μ1, θ, c, emp_opt_cost, emp_cost_std, approx_opt_cost, emp_opt_μ2, approx_opt_μ2, emp_opt_rho, approx_opt_rho, mismatch))
                end                
            end
        end
    end

    return opt_cost_results
end

function mql_comparison(μ1s, μ2, θs, ρs, B, N)
    mql_results = DataFrame(mu1=Float64[], mu2=Float64[], theta=Float64[], rho=Float64[], Empirical_Mean_Q2=Float64[], Empirical_Mean_Q2_Std = Float64[], Approx_Q2 = Float64[], Gap = Float64[], Relative_Gap = Float64[])
    batched_mql_results = DataFrame(rho=Float64[], Batch_Mean_Q2=Float64[], Batch_Mean_Q2_Std = Float64[], Approx_Q2 = Float64[], Gap = Float64[], Relative_Gap = Float64[])
    strt_time = Dates.now()
    progress = 0
    for ρ in ρs
        batch_qls= zeros(Float64, length(μ1s), length(θs), B)
        avg_approx_q2 = 0.0
        for i in 1:length(θs)
            θ = θs[i]
            for j in 1:length(μ1s)
                μ1 = μ1s[j]
                progress += 1
                λ = max_lambda(μ1, μ2, θ, ρ)
                for b in 1:B
                    Random.seed!(random_seed + b)
                    batch_qls[j, i, b] = mean_ql(λ, θ, μ1, μ2, N)
                end
                batch_std = std(batch_qls[j, i, :])/sqrt(B)
                emp_queue2 = mean(batch_qls[j, i, :])

                M = new_M_fn(max_lambda(μ1, μ2, θ, 1), θ, μ1, μ2)
                approx_q2 = M / (1 - ρ)
                
                avg_approx_q2 += approx_q2 / (length(μ1s)*length(θs))

                gap = emp_queue2 - approx_q2
                rel_gap = gap / approx_q2


                push!(mql_results, (μ1, μ2, θ, ρ, emp_queue2, batch_std, approx_q2, gap, rel_gap))

                println("Completed $progress out of $(length(μ1s)*length(θs)*length(ρs)) settings. $((progress/(length(μ1s)*length(θs)*length(ρs)))*100) %")
                println("Elapsed time: ", Dates.now() - strt_time)
                println("----------")
            end
        end
        # After iterating through all μ1 and θ for a given ρ, we can also add a batched summary row for that ρ
        batch_emp_queue2 = mean(batch_qls)
        batch_emp_queue2_std = std(batch_qls)/sqrt(length(μ1s)*length(θs)*B)
        batch_gap = batch_emp_queue2 - avg_approx_q2
        batch_rel_gap = batch_gap / avg_approx_q2
        push!(batched_mql_results, (ρ, batch_emp_queue2, batch_emp_queue2_std, avg_approx_q2, batch_gap, batch_rel_gap))
    end

    return mql_results, batched_mql_results
end
# Param Sets
μ1s = [1.1, 1.2, 1.5, 2, 4]
ρs = [.8, .9, .95, .99]
λs = [0.5, 0.7, 0.9, 1.0, 1.2]
θs = [.1, .3, .5]
μ2 = 1.0

cs = [.5, .75, 1.0, 1.5, 2.0]

μ2_tries = 400

B = 20 # Number of batches to run per parameter set
N = 10^8  # Number of jobs to simulate per batch

# Finding Empirical Distributions
sim_start_time = Dates.now()
println("Starting empirical distribution simulations...")
println("B= $B, N= $N")

emp_distns, batch_emp_distns = genrQ1Q3_emp_distns(μ1s, μ2, θs, ρs, B, N)
# println("Empirical distribution simulations for Q1-Q3 completed.")

# println("Starting empirical distribution simulations for Q4...")
q4_emp_mqls, q4_batch_emp_mqls = genrQ4_mqls(μ1s, θs, λs, μ2_tries, B, N)
# println("Empirical distribution simulations for Q4 completed.")

# Save results to file for later analysis
cur_time = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
println("Results written to JLS files.")
println("Total Simulation Time: ", Dates.now() - sim_start_time)

# Analyzing Results
println("Starting analysis of results...")

mql_comp, batched_mql_comp = mql_comparison(μ1s, μ2, θs, ρs, B, N)
CSV.write(string(raw"Mean_QL_Comparison_Q1Q3_", cur_time, ".csv"), mql_comp)
# CSV.write(string(raw"Batched_Mean_QL_Comparison_Q1Q3_", cur_time, ".csv"), batched_mql_comp)

println("Mean Queue Length Comparison Done")

KS_comp = KS_comparison(emp_distns, batch_emp_distns, B)
corr_comp = corr_comparison(emp_distns, batch_emp_distns, B)
B = 40
N = 10^5
cost_comp = cost_estimation(λs, μ1s, θs, cs, B, N, μ2_tries)

println("Analysis of results completed.")

CSV.write(string(raw"KS_Comparison_Q1Q3_", cur_time, ".csv"), KS_comp)
CSV.write(string(raw"Corr_Comparison_Q1Q3_", cur_time, ".csv"), corr_comp)
CSV.write(string(raw"Opt_Cost_Comparison_Q4_", cur_time, ".csv"), cost_comp)
#println("KS Comparison results for Q3 written to CSV.")


