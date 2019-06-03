import sys
class ATM():
    """
    ATM class 
    """
    
    def __init__(self, bills = [50,20],
                message_error = 'Sua mensagem de erro'):
        '''
        Here the ATM class have some flexibility,
        Like the bills used when withdrawing cash is not hardcoded.
        Also the error message is flexible.
        The next step in the message error is to round the amount to closest value possible to withdraw.
        
        For example:
        if the client request: 125.
        It is not possible (since there are only 50 and 20 bills).
        Therefore the closest value is 120, this could be a nice feature to the user.
        ---------
        INPUT:
        bills: a list or int containing the bills
        message_error: str with custom error message
        '''
        # Creating custom variables
        self.UserIDs = []
        self.requested_values = []
        
        # polymorphic input
        if isinstance(bills, int):
            self.bills = [bills]
        elif isinstance(bills, list):
            self.bills = bills
        else:
            raise Exception('bills input must be a list or a int')
        self.message_error = message_error
        
    def process_request(self, input):
        """
        Process client request
        """
        self.UserIDs.append(input['request']['userID'])
        self.requested_values.append(input['request']['request'])
        
        quantia = input['request']['request']
        res, coins_used = (self.min_bills(quantia))
        response = self.countBills(coins_used, quantia)

        return{'requester': 
               {'userID': input['request']['userID'],
                'requested': quantia},
               'response': response}

#        print(sum(coins_used))
    
    def min_bills(self, value):
        '''
        Decided to use a variation of Dynamic programming algo to find the minimun number of bills.
        This algorithm doesn't suffer of some known issues when using greedy algorithm.
        Also, this variation will have a O(bv), where b is number of bills (2 in this case)
        and v is the amount.
        
        Also I created a temporary list to save the results.
        -----------
        INPUT:
        value: amount to withdraw
        -----------
        OUTPUT:
        minimun_number_of_bills
        bills_used
        '''
        bills_len = len(self.bills)
        bills_used = [0] * (value+1)
        
        # Initialize all values as inf
        reference_table = [sys.maxsize for i in range(value+1)]
        # Base case
        reference_table[0] = 0
        
        for i in range(1, value + 1): # all elements in the ref table
            
            # Go through all coins smaller than i 
            for j in range(bills_len):
                if self.bills[j] <= i:
                
                    sub_res = reference_table[i - self.bills[j]]

                    if sub_res +1 < reference_table[i]:
                        reference_table[i] = sub_res + 1
                        bills_used[i] = self.bills[j]
                        
        return reference_table[j], bills_used

    def countBills(self,bills_used,change):
        '''
        Count the number of bills if possible
        '''
        import collections
        bill = change
        result = []
        while bill > 0:
            current_bill = bills_used[bill]
            if current_bill == 0:
                break
            result.append(current_bill)
            bill -= current_bill
        response = {str(k): v for k,v in collections.Counter(result).items()}
        
        # If no response error message
        if not response:
            response = {'error': self.message_error}
        return response
    
if __name__ == "__main__":

    import json

    # Loads the test file
    # Added a new json file with one additional test
    # caixa_eletronico_amostras_teste_add.json
    input_file = open("caixa_eletronico_amostras_teste.json")
    test_samples = json.load(input_file)

    # Instance proposed class
    atm = ATM()

    # Loops every test sample
    count_success, count_fail = 0, 0
    for idx, sample in enumerate(test_samples):
        sample.get('input')
        input = sample.get("input")
        correct_output = sample.get("output")

        processed_output = atm.process_request(input)

        # Checks if the result is correct
        if processed_output == correct_output:
            count_success += 1
            print("Sample # {} - Success".format(idx))
        else:
            print("Sample # {} - Error".format(idx))
            count_fail += 1
    
    print("You got {} right answers and {} wrong answers".format(count_success, count_fail))