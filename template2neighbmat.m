function [neighbouring_matrix] = template2neighbmat(PATH_TO_TEMPLATE_NEIGHBOURS,nb_elec)

%convert template neighbours from FieldTrip to neighbouring matrix

neighbours = load(PATH_TO_TEMPLATE_NEIGHBOURS);
neighbours = struct2cell(neighbours);
neighbours = neighbours{1};
neighbouring_matrix = zeros(nb_elec,nb_elec);
for i = 1:nb_elec
    idx = [];
    for neighlab = convertCharsToStrings((neighbours(i).neighblabel)')
        tmp = find(strcmp(convertCharsToStrings({neighbours.label}),neighlab{1}));
        idx = [idx tmp];
    end
    neighbouring_matrix(i,idx) = 1;
end