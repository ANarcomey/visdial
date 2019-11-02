raw json data of dialogs can be found in TODO

partitioned data, split into (train1, train2, val) can be found in visdial_[data|params]_partition.[h5|json]

partitioned data of all categories, but with vocabulary indexing specific to each category split, can be found in visdial_partition_all_categories

partitioned data filtered to contain one category, with options for vocabularies of that category and vocabulary from all-category data, can be found in visidal_category_splits_[dialogwise|turnwise]
	v2 of this category data uses a test set filtered by each category, instead of test set of all categories
