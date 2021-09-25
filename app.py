from flask import Flask, jsonify, request, render_template
import numpy as np

app = Flask(__name__)

#Home page
@app.route('/')
def welcome():
	return render_template('index.html')

#more info
@app.route('/moreinfo')
def more_info():
	return render_template('more_info.html')

#disclaimer
@app.route('/disclaimer')
def disclaimer():
	return render_template('disclaimer.html')       

#ovary 
@app.route('/ovary')
def ovary():
	return render_template('ovary.html', data={})

@app.route('/ovary/predict', methods =['POST'])
def ovary_predict():
    # Put all form entries values in a list 
    features = request.form.getlist("Tumor")
    from ovary_utili import my_function, positive, negative, final_labels

    all_features = ['solid', 'solid_necrosis','cystic','unilocular','honeycomb','vegetations', 'papillary','multi','hypointese','calcification','haemo','dark','fat','diffusion','endo']
    postive_features = [1 if feature in features else 0 for feature in all_features]
    postive_features = np.array(postive_features).reshape(15,1)

    result1 =  my_function(pos_weights=positive, neg_weights=negative, some_pos_row=postive_features, labels=final_labels)
    first = {'Type': 'Score'} 


    # Check the output values and retrive the result with html tag based on the value
    if len(features) != 0:
        return render_template('ovary.html', data =  {**first, **result1})

    else:
        return render_template('ovary.html', 
                               data = {**first, **result1})

#thyroid
@app.route('/thyroid')
def thyroid():
    return render_template('thyroid.html')

@app.route('/thyroid/predict', methods =['POST'])
def thyroid_predict():
    from thyroid_util import pos_weights, neg_weights, all_features, labels, thyroid_cal

    user_input = request.form.getlist("Thyroid")

    results = thyroid_cal(pos_weights, neg_weights, user_input, labels)
    
    return render_template('thyroid.html', result = results)

#adult white matter
@app.route('/adultwhitematter')
def whitematter():
    return render_template('whitematter.html', data={})

@app.route('/adultwhitematter/predict', methods =['POST'])
def whitematter_predict():
    from wm_util import  positive, negative, all_features, labels, prevalence, whitemattet_calculator

    user_input = request.form.getlist("wm")
    
    result1 =  whitemattet_calculator(user_input, positive, negative, all_features, labels, prevalence)
    first = {'Type': 'Score'} 
    
    if len(user_input) != 0:
        return render_template('whitematter.html', data =  {**first, **result1})

    else:
        return render_template('whitematter.html', 
                               data = {**first, **result1})

#supretentorial mass
@app.route('/supra')
def supra():
	return render_template('supratentorial.html', data={})

@app.route('/supra/predict', methods =['POST'])
def supratentorial_predict():
    from supra_util import  positive, negative, all_features, labels, prevalence, supra_calculator

    user_input = request.form.getlist("supra")
    
    result1 =  supra_calculator(user_input, positive, negative, all_features, labels, prevalence)
    first = {'Type': 'Score'} 
    
    if len(user_input) != 0:
        return render_template('supratentorial.html', data =  {**first, **result1})

    else:
        return render_template('supratentorial.html', 
                               data = {**first, **result1})

#pediatric white matter
@app.route('/pediatric')
def pead_wm():
	return render_template('pead.html', data={})

@app.route('/pediatric/predict', methods =['POST'])
def pead_wm_predict():
    from peds_util import  positive, negative, all_features, labels, prevalence, peads_calculator

    user_input = request.form.getlist("supra")
    
    result1 =  peads_calculator(user_input, positive, negative, all_features, labels, prevalence)
    first = {'Type': 'Score'} 
        
    if len(user_input) != 0:
        return render_template('pead.html', data =  {**first, **result1})

    else:
        return render_template('pead.html', 
                               data = {**first, **result1})


#sellar masses
@app.route('/sella')
def sella():
	return render_template('sella.html', data={})

@app.route('/sella/predict', methods =['POST'])
def sella_predict():
    from sella_util import  positive, negative, all_features, labels, prevalence, sella_calculator

    user_input = request.form.getlist("supra")
        
    result1 =  sella_calculator(user_input, positive, negative, all_features, labels, prevalence)
    first = {'Type': 'Score'} 
        
    return render_template('sella.html', data = {**first, **result1})

#spine 
@app.route('/spine')
def spine():
	return render_template('spine.html', data={})

@app.route('/spine/predict', methods =['POST'])
def spine_predict():
    # Put all form entries values in a list 
    user_input = request.form.getlist("Thyroid")
    from spine_util import spine_calculator, positive, negative, all_features, labels


    result1 =  spine_calculator(user_input, positive, negative, all_features, labels)
    first = {'Type': 'Score'} 


    # Check the output values and retrive the result with html tag based on the value
    if len(user_input) != 0:
        return render_template('spine.html', data =  {**first, **result1})

    else:
        return render_template('spine.html', 
                               data = {**first, **result1})

###########################################end

		
if __name__ == "__main__":
    app.run()
